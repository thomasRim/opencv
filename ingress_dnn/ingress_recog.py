import pyautogui
import time
from configparser import ConfigParser

import cv2
import numpy as np
import os

from yolo import Yolo

just_test = False
sysPath = os.path.dirname(os.path.abspath(__file__))

# DNN
weights = os.path.join(sysPath, "training/yolov4-ingress_6000.weights")
config = os.path.join(sysPath, "training/yolov4-ingress.cfg")
names = os.path.join(sysPath, "training/images/classes.txt")

# send to yolo
yo = Yolo(weights, config, names)
yo.confidence = 0.2
yo.blobResize = 640 #256,320,416,480,512,544,576,608,640
font = cv2.FONT_HERSHEY_PLAIN

#Read config.ini file
config_object = ConfigParser()
config_object.read("screen.ini")

#Get the USERINFO section
position = config_object["Position"]
x = int(position["x"])
y = int(position["y"])
dimension = config_object["Dimension"]
w = int(dimension["w"])
h = int(dimension["h"])
interaction = config_object["Interaction"]
delay = float(interaction["delay"])
animation = float(interaction["animation"])

## Functions
def operate():
    # pause 1 
    time.sleep(0.5)

    # take screenshot and recognize it
    objects = takeScreenshotAndRecognize()
    
    names = []
    for obj in objects:
        names.append(obj.name)
    print(names)


    # if home_btn - tap it .return
    if existObjectWithName("home_btn", objects):
        clickOnBox(firstObject("home_btn", objects), -40+x, y)
        return

    # if key_btn and any of inventory(inv_reso) - tap key_btn .return
    if existObjectWithName("key_btn", objects):
        clickOnBox(firstObject("key_btn", objects), x, y)
        return

    # if key_btn and none inventory
    if existObjectWithName("key_sel_btn", objects):
        # 3.1 if any reso_filling - tap on it .return
        if existObjectWithName("reso_filling_enl", objects):
            clickOnBox(firstObject("reso_filling_enl", objects), -40+x,y)
            return
        # 3.2 in none reso_filling - drag a bit up .return
        else:
            dragABit()
            return

    # if charge_btn - tap on it .return
    if existObjectWithName("charge_btn", objects) and existObjectWithName("reso_filling_enl", objects):
        clickOnBox(firstObject("charge_btn", objects), x,y)
        return

    # if charge_once_only - tap on it .return
    if existObjectWithName("charge_once_btn", objects):
        clickOnBox(firstObject("charge_once_btn", objects), x,y)
        return

    # if reso_filling and charge_once_all - tap on reso_filling .return
    if existObjectWithName("reso_filling_enl", objects) and existObjectWithName("charge_once_all_btn", objects):
        clickOnBox(firstObject("reso_filling_enl", objects), x,y)
        return

    # if no filling found but charge_once_all - click on it .return
    if existObjectWithName("charge_once_all_btn", objects) and not existObjectWithName("reso_filling_enl", objects):
        clickOnBox(firstObject("charge_once_all_btn", objects), x,y)
        return

    # if close_btn and none reso_filling - tap on it .return
    if existObjectWithName("close_btn", objects) and not existObjectWithName("reso_filling_enl", objects):
        clickOnBox(firstObject("close_btn", objects), x,y)
        return

def takeScreenshotAndRecognize():
    frame = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    frame = frame[y:y+h,x:x+w]
    # image recognition
    yo.detectFrom(frame)

    # for obj in yo.objects:
    #     label = obj.name #str(obj.x) + ", " + str(obj.y)
    #     textColor = (255, 255, 255)
    #     boxColor = (150, 180, 20)
    #     cv2.rectangle(
    #         frame, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), boxColor, 1
    #     )
    #     cv2.putText(frame, label, (obj.x, obj.y - 5), font, 1, textColor, 1)

    # cv2.imshow("detections", frame)
    # cv2.moveWindow("detections", x+w,0)

    return yo.objects

def existObjectWithName(name="", objects=None):
    '''
    name (str) - label for recognized object class
    objects ([Yolo.FoundObject]) - list of recognized objects
    '''
    for obj in objects:
        if obj.name == name :
            print("exist: {}".format(name))
            return True
    print("not exist: {}".format(name))
    return False

def firstObject(name="", objects=None):
    '''
    name (str) - label for recognized object class
    objects ([Yolo.FoundObject]) - list of recognized objects
    '''
    for obj in objects:
        if obj.name == name :
            return obj
    return None

def clickOnBox(object=None, shiftX=0, shiftY=0):
    '''
    object (Yolo.FoundObject) - object to click on
    '''
    
    if object is not None:
        cx = int(object.x+object.width/2+shiftX)
        cy = int(object.y+object.height/2+shiftY)
        print("box {}:{}:{}:{} - click {}:{}".format(object.x, object.y, object.width, object.height, cx, cy))
        pyautogui.leftClick(cx, cy)

def dragABit():
    pyautogui.moveTo(w*0.5, h*0.8)
    pyautogui.drag(0,-h*0.4, button='left',duration=animation*3)

## Start operating
if just_test:
    pyautogui.moveTo(x, y, duration=animation)
    time.sleep(delay)
    pyautogui.moveTo(x+w, y, duration=animation)
    time.sleep(delay)
    pyautogui.moveTo(x+w, y+h, duration=animation)
    time.sleep(delay)
    pyautogui.moveTo(x, y+h, duration=animation)
    pyautogui.moveTo(w*0.5, h*0.8)
    pyautogui.drag(0,-h*0.2, button='left',duration=animation)

else:    
    i = 0
    detect_each_n_frame = 15
    while True:
        if i % detect_each_n_frame == 0:  # each N frame, to fastener
            i = 0

            operate()

        # ch = cv2.waitKey(1)
        # if ch == 27:
        #     break
        i += 1

    cv2.destroyAllWindows()
