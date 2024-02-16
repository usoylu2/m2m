import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import random
import matlab.engine
import numpy as np
from read_files import read, read_mat
import matplotlib.pyplot as plt


fileNames1 = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/stable/sonixone/calib2/ufuk1.rf',
    '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/stable/verasonics/post/calib2.mat']
fileNames2 = ['//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/sonixone/stable/50MHz/glass/ufuk1.rf',
  '//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/verasonics/stable/linear_50MHz/glass_calibration.mat']

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


squared_fourier_transform = np.zeros((Depth, int(len(fileNames1)), int(patch_size / 2)))

for h in range(len(fileNames1)):
    print(fileNames1[h])
    if h == 0:
        rf_np = read(fileNames1[h])
    elif h == 1:
        rf_np = read_mat(fileNames1[h], "calibration")  # Turn this on for stable calibration
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
snr = ((squared_fourier_transform - noise_pow[:, :, np.newaxis]) + 1e-20) / noise_pow[:, :, np.newaxis]
snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
snr1 = np.min(snr, axis=1)

squared_fourier_transform = np.zeros((Depth, int(len(fileNames1)), int(patch_size / 2)))

for h in range(len(fileNames2)):
    print(fileNames2[h])
    if h == 0:
        rf_np = read(fileNames2[h])
    elif h == 1:
        rf_np = read_mat(fileNames2[h], "calibration")  # Turn this on for stable calibration
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
snr = ((squared_fourier_transform - noise_pow[:, :, np.newaxis]) + 1e-20) / noise_pow[:, :, np.newaxis]
snr = np.append(snr, snr[:, :, -1][:, :, np.newaxis], axis=2)
snr2 = np.min(snr, axis=1)

for depth in range(8):
    plt.plot(snr1[depth], label="Calib1")
    plt.plot(snr2[depth], label="Calib2")
    plt.legend()
    plt.show()