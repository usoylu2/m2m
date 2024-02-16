import matlab.engine
import numpy as np
from scipy import interpolate
import scipy.io

# Start a MATLAB engine session
eng = matlab.engine.start_matlab()


def read(filepath):
    """
    Read data from rf file using the MATLAB function. (SonixOne data)

    Args:
    filepath (str): The path to the file to be read.

    Returns:
    np.ndarray: A NumPy array containing the data read from the file.
    """
    rf = eng.RPread(filepath)
    rf_np = np.array(rf._data)
    rf_np = np.reshape(rf_np, rf.size, order="F")
    return rf_np


def read_mat(filename, process):
    """
    Read and interpolate data from a .mat file. (Verasonics data)

    Args:
    filename (str): The path to the .mat file to be read.
    process (str): The type of data ("train" or "calibration").

    Returns:
    np.ndarray: A NumPy array containing the interpolated data from the .mat file.
    """
    x = np.array(range(2864))
    y = np.array(range(128))
    y_new = np.linspace(0, 128, 256)
    if process == "train":
        data_new = np.zeros((256, 2864, 20))
        rep = 20
    elif process == "calibration":
        data_new = np.zeros((256, 2864, 10))
        rep = 10
    else:
        raise TypeError("Wrong process")

    mat = scipy.io.loadmat(filename)
    data = mat['lognorm_mod_after']
    data = np.swapaxes(data, 0, 1)
    for i in range(rep):
        f = interpolate.interp2d(x, y, np.squeeze(data[:, :, i]), kind='linear')
        data_new[:, :, i] = f(x, y_new)

    return np.swapaxes(data_new, 0, 1)


def read_calibration2():
    """
    Read free hand data from calibration phantom 2.

    Returns:
    np.ndarray: A NumPy array containing the concatenated calibration data from the calibration phantom 2.
    """
    data = np.zeros((2864, 256, 1000))
    for i in range(50):
        folder = "//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/PostBeamformed/verasonics/free_calibration/glass"
        ext = ".mat"
        filename = folder + str(i+1) + ext
        data[:, :, i*20:i*20+20] = read_mat(filename, "train")
    return data


def read_calibration1():
    """
    Read free hand data from calibration phantom 1.

    Returns:
    np.ndarray: A NumPy array containing the concatenated calibration data from the calibration phantom 1.
    """
    data = np.zeros((2864, 256, 1000))
    for i in range(50):
        folder = "//bi-isilon-smb.beckman.illinois.edu/oelze/home/usoylu2/dataset/model_security/CIRS/freehand/verasonics/post/calib2_"
        ext = ".mat"
        filename = folder + str(i+1) + ext
        data[:, :, i*20:i*20+20] = read_mat(filename, "train")
    return data


def read_calibration_l11_5():
    """
    Read free hand calibration data from L11-5v.

    Returns:
    np.ndarray: A NumPy array containing the concatenated calibration data.
    """
    x = np.array(range(4296))
    y = np.array(range(128))
    y_new = np.linspace(0, 128, 256)
    data_new = np.zeros((256, 4296, 20))
    data_final = np.zeros((2080, 256, 1000))

    for i in range(50):
        folder = "//172.22.224.234/usoylu2/L11-5v/beamformed/calibration1/calibration"
        ext = ".mat"
        filename = folder + str(i+1) + ext
        mat = scipy.io.loadmat(filename)
        data = mat['lognorm_mod_after']
        data = np.swapaxes(data, 0, 1)
        for j in range(20):
            f = interpolate.interp2d(x, y, np.squeeze(data[:, :, j]), kind='linear')
            data_new[:, :, j] = f(x, y_new)
        data_final[:, :, i*20:i*20+20] = np.swapaxes(data_new, 0, 1)[:2080]
    return data_final
