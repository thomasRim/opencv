from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import os, sys

alpha = 1.0
alpha_max = 500
beta = 0
beta_max = 200
gamma = 1.0
gamma_max = 200


def basicLinearTransform():
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)


    res = cv.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    res = cv.LUT(res, lookUpTable)
    img_corrected = cv.hconcat([img_original, res])
    cv.imshow("Brightness and contrast adjustments", img_corrected)


# def gammaCorrection():
#     ## [changing-contrast-brightness-gamma-correction]
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

#     res = cv.LUT(img_original, lookUpTable)
#     ## [changing-contrast-brightness-gamma-correction]

#     img_gamma_corrected = cv.hconcat([img_original, res])
#     cv.imshow("Gamma correction", img_gamma_corrected)


def on_linear_transform_alpha_trackbar(val):
    global alpha
    alpha = val / 100
    basicLinearTransform()


def on_linear_transform_beta_trackbar(val):
    global beta
    beta = val - 100
    basicLinearTransform()


def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100
    basicLinearTransform()


#
#
#
filename = "saved_pypylon_img.png" #"pypylon_img.png" #"0001.jpg"
sysPath = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(sysPath, filename)

img_original = cv.imread(source_path)

img_corrected = np.empty(
    (img_original.shape[0], img_original.shape[1] * 2, img_original.shape[2]),
    img_original.dtype,
)
img_gamma_corrected = np.empty(
    (img_original.shape[0], img_original.shape[1] * 2, img_original.shape[2]),
    img_original.dtype,
)

img_corrected = cv.hconcat([img_original, img_original])
img_gamma_corrected = cv.hconcat([img_original, img_original])

cv.namedWindow("Brightness and contrast adjustments")
# cv.namedWindow("Gamma correction")

alpha_init = int(alpha * 100)
cv.createTrackbar(
    "Alpha gain (contrast)",
    "Brightness and contrast adjustments",
    alpha_init,
    alpha_max,
    on_linear_transform_alpha_trackbar,
)
beta_init = beta + 100
cv.createTrackbar(
    "Beta bias (brightness)",
    "Brightness and contrast adjustments",
    beta_init,
    beta_max,
    on_linear_transform_beta_trackbar,
)
gamma_init = int(gamma * 100)
cv.createTrackbar(
    "Gamma correction",
     "Brightness and contrast adjustments",
    gamma_init,
    gamma_max,
    on_gamma_correction_trackbar,
)

on_linear_transform_alpha_trackbar(alpha_init)
on_gamma_correction_trackbar(gamma_init)

cv.waitKey()
