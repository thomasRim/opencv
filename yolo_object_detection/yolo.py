import cv2 as cv
import numpy as np
import os
import sys
import utils


class Yolo(object):
    
    def __init__(self):
        super().__init__()
        sysPath = os.path.dirname(os.path.abspath(__file__))
        # Load Yolo
        self.weightPath = os.path.join(sysPath, 'yolov3.weights')
        self.configPath = os.path.join(sysPath, 'yolov3.cfg')
        if not utils.fileExist(self.weightPath) :
            sys.exit(1)
        if not utils.fileExist(self.configPath) :
            sys.exit(1)
        
        self.classes = []
        cocoNamesPath = os.path.join(sysPath, 'coco.names')
        utils.fileExist(cocoNamesPath)
        with open(cocoNamesPath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv.dnn.readNet(self.weightPath, self.configPath)
        # layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def detectFrom(self,img):
        height, width, channels = img.shape

        # Detecting objects
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 608), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        ## Showing informations on the screen
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

        #draw boxes and text
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                textColor = (0,0,187)
                boxColor = (150,180,20)
                cv.rectangle(img, (x, y), (x + w, y + h), boxColor, 1)
                cv.putText(img, label, (x, y - 5), font, 1, textColor, 2)
