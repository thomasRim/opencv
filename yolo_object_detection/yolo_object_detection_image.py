import cv2
import numpy as np
import os
import sys
import yolo
import utils
from datetime import datetime

# Load Yolo
# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

weights = os.path.join(sysPath, 'lib/yolov3-custom.weights')
config = os.path.join(sysPath, 'lib/yolov3-custom.cfg')
names = os.path.join(sysPath, 'lib/custom.names')

# ['sources/washer_1.jpeg', 'sources/washer_2.jpeg', 'sources/washer_3.jpeg', 'sources/washer_4.jpeg', 'sources/washer_5.jpeg', 'sources/washer_6.jpeg']
sources = ['sources/rs-1.jpg', 'sources/rs-2.png', 'sources/rs-3.png']

yo = yolo.Yolo(weights, config, names)

for (i, name) in enumerate(sources):

    # Loading image
    imagePath = os.path.join(sysPath, name)
    if utils.fileExist(imagePath):
        img = cv2.imread(imagePath)
        # img = cv2.resize(img, None, fx=2.4, fy=2.4)

        yo.detectFrom(img)

        # imageName = os.path.join(sysPath, 'result/image_' + str(int(datetime.timestamp(datetime.now()))) + '.jpg')
        # cv2.imwrite(imageName, img)

        cv2.imshow("Image" + str(i), img)


cv2.waitKey(0)
cv2.destroyAllWindows()
