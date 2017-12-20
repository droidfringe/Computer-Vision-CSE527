import cv2
import numpy as np
import matplotlib.pyplot as plt


def CreateLowPassFilteringMask(m, n):
    filteringMask = np.zeros((m, n))
    midRow, midCol = m / 2, n / 2
    maskRows, maskCols = 20, 20
    filteringMask[midRow - maskRows / 2:midRow + maskRows / 2,
                  midCol - maskCols / 2:midCol + maskCols / 2] = 1
    return filteringMask


def CreateHighPassFilteringMask(m, n):
    lowPassfilteringMask = CreateLowPassFilteringMask(m, n)
    highPassfilteringMask = 1 - lowPassfilteringMask
    return highPassfilteringMask


# This function assumes img is represented in uint8 format
def FilterImage(img, filteringMask):
    imgFT = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    imgFTShift = np.fft.fftshift(imgFT)
    filteredFT = np.empty(imgFTShift.shape)
    filteredFT[:, :, 0] = np.multiply(imgFTShift[:, :, 0], filteringMask)
    filteredFT[:, :, 1] = np.multiply(imgFTShift[:, :, 1], filteringMask)
    filteredFTUncentered = np.fft.ifftshift(filteredFT)
    imgFiltered = cv2.idft(
        filteredFTUncentered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    np.clip(imgFiltered, 0, 255, out=imgFiltered)
    return imgFiltered.astype('uint8')


def LowPassFilter(img):
    m, n = img.shape
    filteringMask = CreateLowPassFilteringMask(m, n)
    imgLPF = FilterImage(img, filteringMask)
    return imgLPF


def HighPassFilter(img):
    m, n = img.shape
    filteringMask = CreateHighPassFilteringMask(m, n)
    imgHPF = FilterImage(img, filteringMask)
    return imgHPF


def GetKernel():
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    return gk


# This function assumes imgBlurred is of type np.float and all values are between 0 and 1
def Deconvolution(imgBlurred, kernel):
    #imgFT = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    #kernelFT = cv2.dft(np.float32(kernel), flags = cv2.DFT_COMPLEX_OUTPUT)
    imgBlurredFT = np.fft.fft2(np.float32(imgBlurred))
    kernelFT = np.fft.fft2(np.float32(kernel), imgBlurred.shape)
    imgFT = np.divide(imgBlurredFT, kernelFT)
    img = np.abs(np.fft.ifft2(imgFT))
    np.clip(255 * img, 0, 255, out=img)
    return img.astype('uint8')
