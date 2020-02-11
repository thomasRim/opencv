import cv2
import numpy as np
import os
import sys
from datetime import datetime

# Load Yolo
sysPath = os.path.dirname(os.path.abspath(__file__))
weightPath = os.path.join(sysPath, 'yolov3.weights')
configPath = os.path.join(sysPath, 'yolov3.cfg')
# net = cv2.dnn.readNet(weightPath , configPath)
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
cocoNamesPath = os.path.join(sysPath, 'coco.names')
with open(cocoNamesPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
imagePath = os.path.join(sysPath, 'sources/street_4.jpeg')
if not os.path.isfile(imagePath):
        print("Input image file ", imagePath, " doesn't exist")
        sys.exit(1)
img = cv2.imread(imagePath)
# img = cv2.resize(img, None, fx=2.4, fy=2.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

imageName = os.path.join(sysPath, 'result/image_' + str(int(datetime.timestamp(datetime.now()))) + '.jpg')
# cv2.imshow("Image", img)
cv2.imwrite(imageName, img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()