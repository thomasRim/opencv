import cv2
import numpy as np
import os
import sys
import yolo
import utils

# Load Yolo
# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

weights = os.path.join(sysPath, 'lib/yolov3-custom.weights')
config = os.path.join(sysPath, 'lib/yolov3-custom.cfg')
names = os.path.join(sysPath, 'lib/custom.names')

source = 'sources/color.png'

imagePath = os.path.join(sysPath, source)
img = cv2.imread(imagePath)

height, width, _ = img.shape

# Crop image to get smaller region to detect from.
x = int(width * 0.55)
w = int(width * 0.2)
y = int(height * 0.45)
h = int(height * 0.45)

cropped = img[y: y + h, x: x + w]

alpha = float(4)
beta = int(40)

cropped = cv2.convertScaleAbs(
    cropped, alpha=alpha, beta=beta)

cv2.imwrite("cropped.png", cropped)

# send cropped to yolo
yo = yolo.Yolo(weights, config, names, x, y)
yo.detectFrom(cropped)

# Visualize detected on source image
font = cv2.FONT_HERSHEY_PLAIN

print(str(alpha) + ', ' + str(beta) + ', ' +
      "detected - " + str(len(yo.objects)))
for obj in yo.objects:
    label = obj.name + ', ' + str(obj.x) + ', ' + str(obj.y)
    textColor = (0, 0, 0)
    boxColor = (150, 180, 20)
    cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.width,
                                        obj.y + obj.height), boxColor, 1)
    cv2.putText(img, label, (obj.x, obj.y), font, 1, textColor, 2)

cv2.imwrite("Image.png", img)
