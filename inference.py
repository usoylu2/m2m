import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ResNet, DenseNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statistics
from calibration import TestTimeCalibration

Normalization_FLAG = True
Test_Calibration_FLAG = False
ResNet_flag = False
DenseNet_flag = True

# Get cpu or gpu device for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = nn.Sigmoid()

Filter_length = 51
# filter_aug_test = TestTimeCalibration(filter_length=Filter_length, device=device)
test_images = [2000]
us_images = [2000]
Depth = 9
Batch_size = 2048
Start_pixel = 540
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def extract_all_patch(volume, num):
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


def train_split(vol1, vol2, num):

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


def infer(x_test, y_test, depth_test, mean_data, std_data, PATH):

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

    # # Use Batch Stats in BatchNorm2d layers
    # for module in net.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         module.track_running_stats = False
    #         module.running_var, module.running_mean = None, None
    #         module.eval()

    net.load_state_dict(torch.load(PATH))

    x_test_gpu = torch.from_numpy(x_test[:, np.newaxis, :, :]).float().to(device)
    y_test_gpu = torch.from_numpy(y_test).float().to(device)
    depth_test_gpu = torch.from_numpy(depth_test).to(device)

    # if Test_Calibration_FLAG:
    #     x_test_gpu = filter_aug_test(x_test_gpu, depth_test_gpu)

    # Calculate Mean
    mean_test = torch.mean(x_test_gpu, 0, True)
    std_test = torch.std(x_test_gpu, 0, True)

    # z-score normalization or standardization
    if Normalization_FLAG:
        x_test_gpu = (x_test_gpu-torch.from_numpy(mean_data).float().to(device))/torch.from_numpy(std_data).float().to(device)
        # x_test_gpu = (x_test_gpu-mean_test)/std_test

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
        print(f"Accuracy for class {classname} is: {accuracy}")

    print(f"Average Accuracy is: {total / 2}")
    auc = roc_auc_score(np.hstack(auc_labels), np.hstack(auc_preds))
    print(f"AUC is : {auc}")
    return total/2, auc


if __name__ == '__main__':
    train_vol1 = np.load(f'./train1.npy')
    train_vol2 = np.load(f'./train2.npy')
    test_vol1 = np.load(f'./test1.npy')
    test_vol2 = np.load(f'./test2.npy')
    # test_vol1 = np.load(f'./test1_l11_5.npy')
    # test_vol2 = np.load(f'./test2_l11_5.npy')

    x_train, x_val, y_train, y_val, depth_train, depth_val = train_split(train_vol1, train_vol2, us_images[0])
    x_dev, x_test, y_dev, y_test, depth_dev, depth_test = test_split(test_vol1, test_vol2, test_images[0])
    # folder_name = "train_time_free_calib1"
    folder_name = "nocalibration"
    # folder_name = "test_time"

    c1 = []
    c2 = []

    for i in range(10):
        print(f"Trial:{i}")
        mean_data = np.load(f'./exps/densenet/{folder_name}/mean{i}_calibrated.npy')
        std_data = np.load(f'./exps/densenet/{folder_name}/std{i}_calibrated.npy')
        temp, auc = infer(x_test, y_test, depth_test, mean_data, std_data,
                          f"./exps/densenet/{folder_name}/repetition{i}.pth")
        print("Results:")
        print(f"accuracy {temp}")
        print(f"auc {auc}")
        c1.append(temp)
        c2.append(auc)

    print("Results:")
    print(f"accuracy {c1}; mean: {sum(c1)/len(c1)} and std: {statistics.pstdev(c1)}")
    print(f"auc {c2}; mean: {sum(c2) / len(c2)} and std: {statistics.pstdev(c2)}")
