import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model import ResNet, DenseNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statistics
from calibration import TestTimeCalibration, TrainTimeCalibration
import time
import logging
from torch.utils.tensorboard import SummaryWriter

# Flags for controlling various aspects of the training and testing process
Normalization_FLAG = True
Train_Calibration_FLAG = True
Test_Calibration_FLAG = False
ResNet_flag = True
DenseNet_flag = False

# Folder name for log and model storage
folder_name = "traincalibration"

# Determine the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sigmoid activation function
m = nn.Sigmoid()

# Configure logging for the experiment
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename=f'C:/Users/usoylu2/PycharmProjects/m2m/exps/{folder_name}/train.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Directory for tensorboard logs
log_dir = f"C:/Users/usoylu2/PycharmProjects/m2m/exps/{folder_name}/logs"
writer = SummaryWriter(log_dir)

# Directory for storing trained models
PATH_models = f"C:/Users/usoylu2/PycharmProjects/m2m/exps/{folder_name}/models/"
# Set parameters for training
learning_rate = [1e-5]
test_images = [2000]
us_images = [2000]
epochs = [50]
repetition = 10
# Set parameters for patch extraction
Depth = 9
Batch_size = 2048
Start_pixel = 540
# Initialize Calibration Models
Filter_length = 51
filter_aug_test = TestTimeCalibration(filter_length=Filter_length, device=device)
filter_aug_train = TrainTimeCalibration(probability=1, filter_length=Filter_length, device=device)
# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def extract_all_patch(volume, num):
    """
    Extract patches from a given ultrasound video volume.

    Args:
    volume (numpy.ndarray): The 3D ultrasound video volume from which patches will be extracted.
    num (int): The number of patches to extract.

    Returns:
    patches (numpy.ndarray): An array containing the extracted patches.
    depth_list (numpy.ndarray): An array containing depth information corresponding to each patch.
    """
    indices = np.random.default_rng(seed=0).permutation(volume.shape[2])
    volume = volume[:, :, indices]
    volume = volume[:, :, :num]
    depth, channels, frames = volume.shape

    if channels == 512:
        volume = volume[:, ::2, :]

    start_depth = Start_pixel
    patch_size = 200
    jump = 100

    patches = []
    depth_list = []
    flag = True

    frame_counter = 0
    depth_counter = 0

    while flag:
        for jj in range(9):
            patches.append(volume[start_depth + depth_counter * jump:start_depth + patch_size + depth_counter * jump,
                           10 + 26 * jj:36 + 26 * jj, frame_counter])
            depth_list.append(depth_counter)

        depth_counter += 1
        if depth_counter == Depth:
            frame_counter += 1
            depth_counter = 0

        if start_depth + patch_size + depth_counter * jump >= depth:
            frame_counter += 1
            depth_counter = 0
            patches.pop()
            depth_list.pop()

        if frame_counter == frames:
            flag = False

    return np.array(patches), np.array(depth_list)


def test_function(x_test, y_test, depth_test, mean_data, std_data, PATH):

    if ResNet_flag:
        net = ResNet()
        net = nn.DataParallel(net)
        net.to(device)
    elif DenseNet_flag:
        net = DenseNet()
        net = nn.DataParallel(net)
        net.to(device)
    else:
        raise ValueError("Invalid Network")

    net.load_state_dict(torch.load(PATH))

    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to(device)
    y_test_gpu = torch.from_numpy(y_test).float().to(device)
    depth_test_gpu = torch.from_numpy(depth_test).to(device)

    if Test_Calibration_FLAG:
        x_test_gpu = filter_aug_test(x_test_gpu, depth_test_gpu)

    # Calculate Mean
    mean_test = torch.mean(x_test_gpu, 0, True)
    std_test = torch.std(x_test_gpu, 0, True)

    # z-score normalization or standardization
    if Normalization_FLAG:
        x_test_gpu = (x_test_gpu-mean_test)/std_test

    dataset = TensorDataset(x_test_gpu, y_test_gpu, depth_test_gpu)
    test_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=True)

    # prepare to count predictions for each class
    classes = ["phantom1", "phantom2"]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    data_matrix = np.zeros((2, 2))
    depth_matrix = np.zeros((Depth, 3))
    net.eval()
    auc_labels = []
    auc_preds = []
    # again no gradients needed

    # filter_aug = Firwin_test()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels, depth = data

            outputs = net(inputs)

            auc_labels.append(labels.cpu().detach().numpy())
            auc_preds.append(m(outputs)[:, 0].cpu().detach().numpy())

            predictions = ((m(outputs) > 0.5) * 1)[:, 0]
            if predictions.shape[0] != labels.shape[0]:
                raise ValueError("Error in label shape")
            # collect the correct predictions for each class
            for index in range(predictions.shape[0]):
                if int(labels[index]) == int(predictions[index]):
                    correct_pred[classes[int(labels[index])]] += 1
                    depth_matrix[int(depth[index]), 0] = depth_matrix[int(depth[index]), 0] + 1
                else:
                    depth_matrix[int(depth[index]), 1] = depth_matrix[int(depth[index]), 1] + 1
                total_pred[classes[int(labels[index])]] += 1
                if int(labels[index]) == 0:
                    data_matrix[0, int(predictions[index])] += 1
                elif int(labels[index]) == 1:
                    data_matrix[1, int(predictions[index])] += 1
                # elif label == 2:
                #     data_matrix[2, prediction] += 1
                # elif label == 3:
                #     data_matrix[3, prediction] += 1

    # print accuracy for each class
    total = 0
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        total = total + accuracy
        logger.info(f"Accuracy for class {classname} is: {accuracy}")

    logger.info(f"Average Accuracy is: {total / 2}")
    auc = roc_auc_score(np.hstack(auc_labels), np.hstack(auc_preds))
    logger.info(f"AUC is : {auc}")
    return total/2, auc


def train_function(x_train, x_valid, y_train, y_valid, depth_train, depth_valid, PATH, epoch_num, LR):

    x_train_gpu = torch.from_numpy(x_train[:, np.newaxis, :, :]).float().to(device)
    y_train_gpu = torch.from_numpy(y_train).float().to(device)
    depth_train_gpu = torch.from_numpy(depth_train).to(device)

    x_valid_gpu = torch.from_numpy(x_valid[:, np.newaxis, :, :]).float().to(device)
    y_valid_gpu = torch.from_numpy(y_valid).float().to(device)
    depth_valid_gpu = torch.from_numpy(depth_valid).to(device)

    if Train_Calibration_FLAG:
        x_train_gpu = filter_aug_train(x_train_gpu, depth_train_gpu)
    x_valid_gpu = filter_aug_train(x_valid_gpu, depth_valid_gpu)

    # Calculate Mean
    mean_data = torch.mean(x_train_gpu, 0, True)
    std_data = torch.std(x_train_gpu, 0, True)

    mean_valid = torch.mean(x_valid_gpu, 0, True)
    std_valid = torch.std(x_valid_gpu, 0, True)

    # z-score normalization or standardization
    if Normalization_FLAG:
        x_train_gpu = (x_train_gpu-mean_data)/std_data
        x_valid_gpu = (x_valid_gpu-mean_valid)/std_valid

    dataset = TensorDataset(x_train_gpu, y_train_gpu, depth_train_gpu)
    train_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=False)

    dataset = TensorDataset(x_valid_gpu, y_valid_gpu, depth_valid_gpu)
    valid_loader = DataLoader(dataset, batch_size=Batch_size, pin_memory=False, shuffle=False)

    if ResNet_flag:
        net = ResNet()
        net = nn.DataParallel(net)
        net.to(device)
    elif DenseNet_flag:
        net = DenseNet()
        net = nn.DataParallel(net)
        net.to(device)
    else:
        raise ValueError("Invalid Network")

    parameter_number = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters:{parameter_number}")

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_epoch = []
    accuracies = []
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 40], gamma=0.5)
    hflipper = T.RandomHorizontalFlip(p=0.5)
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(epoch_num):# loop over the dataset multiple times
        start_time = time.time()

        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, depth = data

            #Data Augmentation  https://pytorch.org/vision/stable/transforms.html
            inputs = hflipper(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                outputs = net(inputs)
                loss = criterion(outputs[:, 0], labels)

            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            # print statistics
            # loss_epoch.append(loss.item())
            running_loss += loss.item()
        # scheduler.step()
        loss_epoch.append(running_loss)
        logger.info(f'{epoch+1} loss: {running_loss}')

        if (epoch+1) % 1 == 0:
            classes = ["phantom1", "phantom2"]
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            data_matrix = np.zeros((2, 2))
            net.eval()

            # again no gradients needed
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels, depth = data
                    outputs = net(inputs)
                    predictions = ((m(outputs) > 0.5)*1)[:, 0]
                    if predictions.shape[0] != labels.shape[0]:
                        print("Error in label shape")
                    # collect the correct predictions for each class
                    for index in range(predictions.shape[0]):
                        if int(labels[index]) == int(predictions[index]):
                            correct_pred[classes[int(labels[index])]] += 1
                        total_pred[classes[int(labels[index])]] += 1
                        if int(labels[index]) == 0:
                            data_matrix[0, int(predictions[index])] += 1
                        elif int(labels[index]) == 1:
                            data_matrix[1, int(predictions[index])] += 1
                        # elif label == 2:
                        #     data_matrix[2, prediction] += 1
                        # elif label == 3:
                        #     data_matrix[3, prediction] += 1
            # print accuracy for each class
            total = 0
            logger.info(f"Epoch: {epoch}")
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                total = total + accuracy
                logger.info(f"Accuracy for class {classname} is: {accuracy}")
            logger.info(f"Average Accuracy is: {total / 2}")
            accuracies.append(total/2)
            writer.add_scalar("Accuracy/valid", total/2, epoch)
        logger.info(f"Execution time is {time.time() - start_time} seconds")
        torch.save(net.state_dict(), PATH_models+f"epoch{epoch}.pth")
    logger.info('Finished Training')

    torch.save(net.state_dict(), PATH)

    return mean_data.cpu().detach().numpy()[0], std_data.cpu().detach().numpy()[0], accuracies


def train_split(vol1, vol2, num):
    """
    Split and shuffle the patches from two volumes for training.

    Args:
    vol1 (numpy.ndarray): The first volume containing patches.
    vol2 (numpy.ndarray): The second volume containing patches.
    num (int): The number of patches to use from each volume.

    Returns:
    x_train (numpy.ndarray): Training data containing patches.
    x_dev (numpy.ndarray): Development data containing patches.
    y_train (numpy.ndarray): Training labels (0 for vol1, 1 for vol2).
    y_dev (numpy.ndarray): Development labels (0 for vol1, 1 for vol2).
    depth_train (numpy.ndarray): Training depth information.
    depth_dev (numpy.ndarray): Development depth information.
    """
    num = int(num//2)

    class1, depth1 = extract_all_patch(vol1, num)
    class2, depth2 = extract_all_patch(vol2, num)

    x = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1))
    depth = np.concatenate((depth1, depth2), axis=0)

    indices = np.random.default_rng(seed=0).permutation(x.shape[0])
    x = x[indices, :, :]
    depth = depth[indices]
    y = y[indices]

    x_train, x_dev, y_train, y_dev, depth_train, depth_dev = train_test_split(x, y, depth,
                                                                                 test_size=0.2, random_state=42)
    return x_train, x_dev, y_train, y_dev, depth_train, depth_dev


def test_split(vol1, vol2, num):
    """
    Split and shuffle the patches from two volumes for testing.

    Args:
    vol1 (numpy.ndarray): The first volume containing patches.
    vol2 (numpy.ndarray): The second volume containing patches.
    num (int): The number of patches to use from each volume.

    Returns:
    x_test1 (numpy.ndarray): Test data1 containing patches.
    x_test2 (numpy.ndarray): Test data2 containing patches.
    y_test1 (numpy.ndarray): Testing labels1 (0 for vol1, 1 for vol2).
    y_test2 (numpy.ndarray): Testing labels2 (0 for vol1, 1 for vol2).
    depth_test1 (numpy.ndarray): Testing depth1 information.
    depth_test2 (numpy.ndarray): Testing depth2 information.
    """
    num = int(num//2)

    class1, depth1 = extract_all_patch(vol1, num)
    class2, depth2 = extract_all_patch(vol2, num)

    x = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((np.zeros(class1.shape[0]), np.zeros(class2.shape[0]) + 1))
    depth = np.concatenate((depth1, depth2), axis=0)

    indices = np.random.default_rng(seed=0).permutation(x.shape[0])
    x = x[indices, :, :]
    depth = depth[indices]
    y = y[indices]

    x_test1, x_test2, y_test1, y_test2, depth_test1, depth_test2 = train_test_split(x, y, depth,
                                                                                 test_size=0.5, random_state=42)
    return x_test1, x_test2, y_test1, y_test2, depth_test1, depth_test2


if __name__ == '__main__':
    logger.info("Training Started")
    logger.info(f"Using {device} device")

    # Loop through different experiment configurations
    for train_num, epoch, test_num, LR in zip(us_images, epochs, test_images, learning_rate):
        logger.info(f"US image number:{train_num}")
        logger.info(f"Learning Rate: {LR}")
        c1 = []  # List to store accuracy results
        c2 = []  # List to store AUC results

        # Repeat the experiment for multiple trials
        for i in range(repetition):
            logger.info(f"Trial:{i+1}")

            # Load training and test data
            train_vol1 = np.load(f'./train1.npy')
            train_vol2 = np.load(f'./train2.npy')
            test_vol1 = np.load(f'./test1.npy')
            test_vol2 = np.load(f'./test2.npy')

            # Split data into training and validation sets
            x_train, x_dev, y_train, y_dev, depth_train, depth_dev = train_split(train_vol1, train_vol2, train_num)
            x_test1, x_test2, y_test1, y_test2, depth_test1, depth_test2 = test_split(test_vol1, test_vol2, test_num)
            logger.info(f"{x_train.shape}, {x_dev.shape}, {y_train.shape}, {y_dev.shape}")

            # Train the model and obtain mean and standard deviation of the data
            mean_data, std_data, acc = train_function(x_train, x_dev, y_train, y_dev, depth_train, depth_dev,
                                                      f"./exps/{folder_name}/repetition{i}.pth", epoch, LR)

            # Save the mean and standard deviation data
            np.save(f'./exps/{folder_name}/mean{i}.npy', mean_data)
            np.save(f'./exps/{folder_name}/std{i}.npy', std_data)

            logger.info("Testing:")
            logger.info(f"{x_test1.shape}, {y_test1.shape}, {x_test2.shape}, {y_test2.shape}")

            # Test the model and calculate AUC
            temp, auc = test_function(x_test2, y_test2, depth_test2, mean_data, std_data,
                                      f"./exps/{folder_name}/repetition{i}.pth")
            logger.info(f"Test results - Accuracy: {temp}, AUC: {auc}")
            c1.append(temp)
            c2.append(auc)

        # Report the results of all trials for the current experiment configuration
        logger.info("Results:")
        logger.info(f"accuracy {c1}; mean: {sum(c1)/len(c1)} and std: {statistics.pstdev(c1)}")
        logger.info(f"auc {c2}; mean: {sum(c2) / len(c2)} and std: {statistics.pstdev(c2)}")
