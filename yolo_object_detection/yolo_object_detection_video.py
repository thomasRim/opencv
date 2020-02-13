import cv2 as cv
import numpy as np
import os
import sys

def fileExist(filePath):
    if not os.path.isfile(filePath):
        print("Input file ", filePath, " doesn't exist")
        sys.exit(1)

def folderExist(filePath):
    if not os.path.isdir(filePath):
        print("Input file ", filePath, " doesn't exist")
        return False
    else:
        return True

# Load Yolo
sysPath = os.path.dirname(os.path.abspath(__file__))
weightPath = os.path.join(sysPath, 'yolov3.weights')
configPath = os.path.join(sysPath, 'yolov3.cfg')
fileExist(weightPath)
fileExist(configPath)

# Net
# net = cv.dnn.readNet(weightPath, configPath)
net = cv.dnn.readNetFromDarknet(configPath, weightPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# classes
classes = []
cocoNamesPath = os.path.join(sysPath, 'coco.names')
fileExist(cocoNamesPath)
with open(cocoNamesPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detectFrom(img):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.0005, (320, 320), (0, 0, 0), True, crop=False)

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

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            textColor = (0,0,187)
            boxColor = (150,180,20)
            cv.rectangle(img, (x, y), (x + w, y + h), boxColor, 1)
            cv.putText(img, label, (x, y + 30), font, 1, textColor, 2)


# Video
videoName = 'vtest.avi'
source = os.path.join(os.path.join(sysPath, 'sources'),videoName)
fileExist(source)
cap = cv.VideoCapture(source)

ret, img = cap.read()
height, width, channels = img.shape

resultFolder = os.path.join(sysPath, 'result')
if not folderExist(resultFolder):
    os.mkdir(resultFolder)
outSource = os.path.join(resultFolder,videoName)
capWrite = cv.VideoWriter(outSource, int(cap.get(cv.CAP_PROP_FOURCC)), int(cap.get(cv.CAP_PROP_FPS)), (width, height) )

while True:
    ret, img = cap.read()
    detectFrom(img)

    # cv.imshow('Video', img)
    capWrite.write(img)
    ch = cv.waitKey(1)
    if ch == 27:
        break

capWrite.release()
cap.release()
# cv.destroyAllWindows()