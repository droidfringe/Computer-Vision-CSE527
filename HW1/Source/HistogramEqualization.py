import cv2
import numpy as np
import matplotlib.pyplot as plt


# Ref - http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
def HistogramEqualizeOneChannel(img, DEBUG=False):
    m, n = np.shape(img)
    imgHistogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = np.cumsum(imgHistogram)
    if DEBUG:
        plt.bar(range(len(imgHistogram)), imgHistogram)
        plt.show()
        plt.bar(range(len(cdf)), cdf)
        plt.show()
    cdfNormalized = np.divide(cdf, m * n)
    imgNormalized = cdfNormalized[img]
    imgNormalized = (255 * imgNormalized).astype(np.uint8)
    # For verification
    if DEBUG:
        imgNormalizedHistogram = cv2.calcHist([imgNormalized], [0], None, [256],
                                              [0, 256])
        cdfNormalized = np.cumsum(imgNormalizedHistogram)
        plt.bar(range(len(imgNormalizedHistogram)), imgNormalizedHistogram)
        plt.show()
        plt.bar(range(len(cdfNormalized)), cdfNormalized)
        plt.show()
        cv2.imshow('Original', img)
        cv2.imshow('Equalized', imgNormalized)
        cv2.waitKey(0)
    return imgNormalized


# Reference - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
def HistogramEqualizationOfImage(img, DEBUG=False):
    b, g, r = cv2.split(img)
    bEqualized = HistogramEqualizeOneChannel(b, DEBUG)
    gEqualized = HistogramEqualizeOneChannel(g, DEBUG)
    rEqualized = HistogramEqualizeOneChannel(r, DEBUG)
    imgEqualized = cv2.merge((bEqualized, gEqualized, rEqualized))
    return imgEqualized


def OpenCVHistogramEqualization(img):
    b, g, r = cv2.split(img)
    bEqualized = cv2.equalizeHist(b)
    gEqualized = cv2.equalizeHist(g)
    rEqualized = cv2.equalizeHist(r)
    imgEqualized = cv2.merge((bEqualized, gEqualized, rEqualized))
    return imgEqualized


if __name__ == '__main__':
    img = cv2.imread('input1.jpg', cv2.IMREAD_COLOR)
    print(img.shape)
    histEqImg = HistogramEqualizationOfImage(img)
    cv2.imshow('OriginalImage', img)
    cv2.imshow('HistogramEqualizedImage', histEqImg)
    inbuiltHistEq = OpenCVHistogramEqualization(img)
    cv2.imshow('HistogramEqualizedImageOpenCV', inbuiltHistEq)
    print(np.max(abs(inbuiltHistEq - histEqImg)))
    cv2.imwrite('output1.png', histEqImg)
    cv2.waitKey(0)
