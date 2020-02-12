import cv2 as cv
import numpy as np
import os

sysPath = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(sysPath, 'vtest.avi')
cap = cv.VideoCapture(source)

while True:
    ret, img = cap.read()
    cv.imshow('Video', img)
    ch = cv.waitKey(1)
    if ch == 27:
        break

cv.destroyAllWindows()