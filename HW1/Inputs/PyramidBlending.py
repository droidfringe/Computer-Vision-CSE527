import cv2
import numpy as np
import matplotlib.pyplot as plt


def GaussianPyramid(img, numLevels):
    gaussianPyramid = [img.astype('float32')]
    for i in range(numLevels):
        img = cv2.pyrDown(img)
        gaussianPyramid.append(img.astype('float32'))
    return gaussianPyramid


def LaplacianPyramid(gaussianPyramid):
    numLevels = len(gaussianPyramid)
    laplacianPyramid = []
    for i in range(numLevels - 1):
        laplacianCurr = np.subtract(gaussianPyramid[i],(cv2.pyrUp(gaussianPyramid[i + 1])).astype('float32'))
        laplacianPyramid.append(laplacianCurr)
    laplacianPyramid.append(gaussianPyramid[-1])
    return laplacianPyramid


def ReconstructImage(laplacianPyramid):
    numLevels = len(laplacianPyramid)
    currentRecImg = laplacianPyramid[-1]
    for i in range(numLevels - 2, -1, -1):
        currentRecImgUpsampled = cv2.pyrUp(currentRecImg)
        currentRecImg = np.add(currentRecImgUpsampled, laplacianPyramid[i])
    np.clip(currentRecImg, 0, 255, out=currentRecImg)
    return currentRecImg.astype('uint8')


def BlendImages(img1, img2):
    numLevels = 5
    gp1 = GaussianPyramid(img1, numLevels)
    lp1 = LaplacianPyramid(gp1)
    gp2 = GaussianPyramid(img2, numLevels)
    lp2 = LaplacianPyramid(gp2)
    lpMerged = []
    for i in range(len(lp1)):
        mask1 = np.zeros(lp1[i].shape)
        mask2 = np.zeros(lp1[i].shape)
        mask1[:, 0:lp1[i].shape[1] / 2, :] = 1
        mask2[:, lp1[i].shape[1] / 2:, :] = 1
        currLevel = np.add(
            np.multiply(lp1[i], mask1.astype('float32')),
            np.multiply(lp2[i], mask2.astype('float32')))
        lpMerged.append(currLevel)
    imgBlended = ReconstructImage(lpMerged)
    return imgBlended


if __name__ == '__main__':
    img_in1 = cv2.imread('input3A.jpg', cv2.IMREAD_COLOR)
    img_in2 = cv2.imread('input3B.jpg', cv2.IMREAD_COLOR)
    m1, n1, c1 = img_in1.shape
    m2, n2, c2 = img_in2.shape
    img_in1 = img_in1[:, :m1, :]
    img_in2 = img_in2[:m1, :m1, :]
    img_out = BlendImages(img_in1, img_in2)
    cv2.imshow('Blending',img_out)
    cv2.waitKey(0)
