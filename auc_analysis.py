# 1. Load the no calibration model
# 2. Load test set and select a number of frames
# 3. determine an optimal threshold

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import ResNet, DenseNet
from sklearn.model_selection import train_test_split
from calibration import TestTimeCalibration, TrainTimeCalibration
import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import statistics

# Flags for controlling various aspects of the training and testing process
Normalization_FLAG = True
ResNet_flag = False
DenseNet_flag = True

# Folder name for log and model storage
folder_name = "threshold"

# Determine the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sigmoid activation function
m = nn.Sigmoid()

# Configure logging for the experiment
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename=f'C:/Users/usoylu2/PycharmProjects/m2m/revision/exps/{folder_name}/train.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Set parameters for patch extraction
Depth = 9
Batch_size = 2048
Start_pixel = 540

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


def test_function(net, x_test, y_test, depth_test):
    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to(device)
    y_test_gpu = torch.from_numpy(y_test).float().to(device)
    depth_test_gpu = torch.from_numpy(depth_test).to(device)

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
    return np.hstack(auc_labels), np.hstack(auc_preds)


def calculate_accuracy(predictions, labels):
    logger.info(f"Accuracy: {np.mean(predictions.copy() == labels.copy())}")
    return np.mean(predictions.copy() == labels.copy())


if __name__ == '__main__':
    c1, c2 = [], []

    for i in range(10):
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

        net.load_state_dict(torch.load(f"./exps/{folder_name}/repetition{i}.pth"))

        test_vol1 = np.load(f'./test1.npy')
        test_vol2 = np.load(f'./test2.npy')
        x_test1, x_test2, y_test1, y_test2, depth_test1, depth_test2 = test_split(test_vol1, test_vol2, 200)
        logger.info(f"{x_test1.shape}, {x_test2.shape}, {y_test1.shape}, {y_test2.shape}")

        labels, preds = test_function(net, x_test1, y_test1, depth_test1)

        logger.info(f"AUC:{roc_auc_score(labels, preds)}")
        fpr, tpr, thresholds = roc_curve(labels, preds)

        # # Plot ROC curve
        # plt.plot(fpr, tpr)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.grid(True)
        # plt.show()

        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        logger.info(f"Threshold at 0.5")
        _ = calculate_accuracy(np.array((preds.copy() > 0.5) * 1), labels)

        logger.info(f"Optimal Threshold via ROC analysis: {optimal_threshold}")
        _ = calculate_accuracy(np.array((preds.copy() > optimal_threshold) * 1), labels)

        labels, preds = test_function(net, x_test2, y_test2, depth_test2)
        tmp = calculate_accuracy(np.array((preds.copy() > optimal_threshold) * 1), labels)
        c1.append(tmp)
        c2.append(roc_auc_score(labels, preds))

    # Report the results of all trials for the current experiment configuration
    logger.info("Results:")
    logger.info(f"accuracy {c1}; mean: {sum(c1) / len(c1)} and std: {statistics.pstdev(c1)}")
    logger.info(f"auc {c2}; mean: {sum(c2) / len(c2)} and std: {statistics.pstdev(c2)}")