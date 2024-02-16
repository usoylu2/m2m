import numpy as np
import matplotlib.pyplot as plt
from read_files import read, read_calibration_l11_5
from scipy.signal import hilbert

x = read_calibration_l11_5()
print(x.shape)
# test_vol1 = np.load(f'./test1.npy')
# test_vol2 = np.load(f'./test2.npy')
# print(test_vol1.shape, test_vol2.shape)
#
# plt.imshow(20*np.log10(np.abs(hilbert(test_vol2[500:2000, :, 5], axis=0))),
#            aspect='auto', cmap='gray', vmin=0, vmax=90)
# plt.title("Test Data")
# plt.colorbar()
# plt.show()
