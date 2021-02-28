import cv2
import numpy as np
import os
import sys
import time
import yolo
import utils
from datetime import datetime

# Load Yolo

## constants that should be changed from params (future)

lib_folder = "lib"
yolo_weight = "yolov4.weights"
yolo_config = "yolov4.cfg"
yolo_names = "custom.names"

source_folder = "sources"
source_file_name = "red.png"

# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

# YOLO
weights = os.path.join(sysPath, os.path.join(lib_folder,yolo_weight))
config = os.path.join(sysPath, os.path.join(lib_folder,yolo_config))
names = os.path.join(sysPath, os.path.join(lib_folder,yolo_names))

imagePath = os.path.join(sysPath, os.path.join(source_folder,source_file_name))
img = cv2.imread(imagePath)

height, width, _ = img.shape
pre_scale = 1
post_scale = 1

# Crop image to get smaller region to detect from.
x, y, w, h = (
    0,
    0,
    width,
    height,
)  # (int(width * 0.48), int(height * 0.45), int(width * 0.2), int(height * 0.45))

img = img[y : y + h, x : x + w]

img = cv2.resize(img, (int(pre_scale * width), int(pre_scale * height)))

# send to yolo
yo = yolo.Yolo(weights, config, names, x, y)
yo.confidence = 0.2

start = time.time()
yo.detectFrom(img)
recog_end = time.time()

# Visualize detected on source image
font = cv2.FONT_HERSHEY_PLAIN

for obj in yo.objects:
    label = str(obj.x) + ", " + str(obj.y)
    textColor = (0, 0, 0)
    boxColor = (150, 180, 20)
    cv2.rectangle(
        img, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), boxColor, 1
    )
    cv2.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)

draw_end = time.time()

img = cv2.resize(img, (int(post_scale * width), int(post_scale * height)))

cv2.imshow("Image", img)

t1 = recog_end - start
t2 = draw_end - recog_end

print("recognition time: {:5f}, draw time:{:5f}".format(t1, t2))

cv2.waitKey(0)
cv2.destroyAllWindows()
