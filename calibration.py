import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import random
import matlab.engine
import numpy as np
from read_files import read, read_mat, read_calibration1, read_calibration_l11_5

# Start a MATLAB engine session
eng = matlab.engine.start_matlab()
# Set parameters for patch extraction
start_depth = 540
patch_size = 200
jump = 100
Depth = 9
# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Select the filenames that point the calibration files:
# -first one is for Calibration Phantom 1 & Stable Acq
# -second one is for Calibration Phantom 1 & Freehand Acq
# -third one is for Calibration Phantom 2 & Freehand Acq
# -fourth one is for Calibration Phantom 2 & Stable Acq

# fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/stable/sonixone/calib2/ufuk1.rf',
#     '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/stable/verasonics/post/calib2.mat']
fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/freehand/sonixone/calib2/ufuk1.rf',
    '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/freehand/verasonics/post/calib2_1.mat']
# fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/sonixone/free_calibration/glass/ufuk1.rf',
#              '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/verasonics/free_calibration/glass1.mat']
# fileNames = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/sonixone/stable/50MHz/glass/ufuk1.rf',
#   '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/verasonics/stable/linear_50MHz/glass_calibration.mat']


class TestTimeCalibration(nn.Module):
    def __init__(self, device="cuda", fs=4e7, filter_length=51):
        """
        Initialize the TestTimeCalibration model for test-time calibration.
        """
        super(TestTimeCalibration, self).__init__()

        sampling_frequency = 40000000
        values = np.arange(int(patch_size / 2))
        time_period = patch_size / sampling_frequency
        frequencies = values / time_period
        squared_fourier_transform = np.zeros((Depth, int(len(fileNames)), int(patch_size / 2)))

        for h in range(len(fileNames)):
            print(fileNames[h])
            if h == 0:
                rf_np = read(fileNames[h])
            elif h == 1:
                # rf_np = read_calibration1()  # Turn this on for free hand calibration for L9-4
                rf_np = read_mat(fileNames[h], "calibration")  # Turn this on for stable calibration
                # rf_np = read_calibration_l11_5() # Turn this on for free hand calibration for L11-5
            else:
                raise TypeError("Wrong filenames")
            for depth_index in range(Depth):
                for j in range(rf_np.shape[2]):
                    for i in range(rf_np.shape[1]):
                        amplitude = rf_np[start_depth + depth_index * jump: start_depth +
                                          patch_size + depth_index * jump, i, j]
                        # Frequency domain representation
                        fourier_transform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                        fourier_transform = fourier_transform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                        squared_fourier_transform[depth_index, h, :] = squared_fourier_transform[depth_index, h, :] \
                                                                       + pow(abs(fourier_transform), 2)
                squared_fourier_transform[depth_index, h] = squared_fourier_transform[depth_index, h] / \
                                                            (rf_np.shape[2] * rf_np.shape[1])

        noise_pow = np.min(squared_fourier_transform, axis=2)
        snr = ((squared_fourier_transform-noise_pow[:, :, np.newaxis])+1e-20)/noise_pow[:, :, np.newaxis]
        snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
        snr = np.min(snr, axis=1)

        h_abs = np.zeros((Depth, patch_size // 2 + 1))
        for depth_index in range(Depth):
            temp = np.sqrt(squared_fourier_transform[depth_index, 1] /
                           (squared_fourier_transform[depth_index, 0] + 1e-20))
            h_abs[depth_index] = np.append(temp, temp[-1])
        h_abs = np.abs(h_abs)
        wiener_res = h_abs / (h_abs * h_abs + 1 / snr)

        self.desired = wiener_res
        self.bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.fs = fs
        self.kernel = []

        for depth_index in range(Depth):
            temp = torch.from_numpy(signal.firwin2(filter_length, self.bands, self.desired[depth_index],
                                                   fs=self.fs)[np.newaxis, np.newaxis, :, np.newaxis])
            self.kernel.append(temp.type(torch.FloatTensor).to(device))

    def forward(self, x, depth):
        for i in range(x.shape[0]):
            x[i, :, :, :] = F.conv2d(x[i][np.newaxis, :, :, :], self.kernel[depth[i]], padding='same')
        return x


class TrainTimeCalibration(nn.Module):
    def __init__(self, device="cuda", fs=4e7, probability=1, filter_length=51):
        """
        Initialize the TrainTimeCalibration model for train-time calibration.
        """
        super(TrainTimeCalibration, self).__init__()

        sampling_frequency = 40000000
        values = np.arange(int(patch_size / 2))
        time_period = patch_size / sampling_frequency
        frequencies = values / time_period
        squared_fourier_transform = np.zeros((Depth, int(len(fileNames)), int(patch_size / 2)))

        for h in range(len(fileNames)):
            print(fileNames[h])
            if h == 0:
                rf_np = read(fileNames[h])
            elif h == 1:
                rf_np = read_mat(fileNames[h], "calibration")  # Turn this on for stable calibration
                # rf_np = read_calibration1()  # Turn this on for free hand calibration
                # rf_np = read_calibration_l11_5() # Turn this on for free hand calibration for L11-5
            else:
                raise TypeError("Wrong filenames")
            for depth_index in range(Depth):
                for j in range(rf_np.shape[2]):
                    for i in range(rf_np.shape[1]):
                        amplitude = rf_np[start_depth + depth_index * jump: start_depth + patch_size + depth_index * jump, i, j]
                        # Frequency domain representation
                        fourier_transform = np.fft.fft(amplitude) / len(amplitude)  # Normalize amplitude
                        fourier_transform = fourier_transform[range(int(len(amplitude) / 2))]  # Exclude sampling frequency
                        squared_fourier_transform[depth_index, h, :] = squared_fourier_transform[depth_index, h, :] \
                                                                       + pow(abs(fourier_transform), 2)
                squared_fourier_transform[depth_index, h] = squared_fourier_transform[depth_index, h] / \
                                                            (rf_np.shape[2] * rf_np.shape[1])

        noise_pow = np.min(squared_fourier_transform, axis=2)
        snr = ((squared_fourier_transform-noise_pow[:, :, np.newaxis])+1e-20)/noise_pow[:, :, np.newaxis]
        snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
        snr = np.min(snr, axis=1)

        h_abs = np.zeros((Depth, patch_size // 2 + 1))
        for depth_index in range(Depth):
            temp = np.sqrt(squared_fourier_transform[depth_index, 0] /
                           (squared_fourier_transform[depth_index, 1] + 1e-20))
            h_abs[depth_index] = np.append(temp, temp[-1])
        h_abs = np.abs(h_abs)
        wiener_res = h_abs / (h_abs * h_abs + 1 / snr)
        self.p = probability
        self.desired = wiener_res
        self.bands = np.append(frequencies, frequencies[-1] + frequencies[1])
        self.fs = fs
        self.kernel = []

        for depth_index in range(Depth):
            temp = torch.from_numpy(signal.firwin2(filter_length, self.bands, self.desired[depth_index],
                                                   fs=self.fs)[np.newaxis, np.newaxis, :, np.newaxis])
            self.kernel.append(temp.type(torch.FloatTensor).to(device))

    def forward(self, x, depth):
        if random.uniform(0, 1) < self.p:
            for i in range(x.shape[0]):
                x[i, :, :, :] = F.conv2d(x[i][np.newaxis, :, :, :], self.kernel[depth[i]], padding="same")
            return x
        else:
            return x
