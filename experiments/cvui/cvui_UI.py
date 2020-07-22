import numpy as np
import cv2
import cvui
import os
from yolo import Yolo

WINDOW_NAME = 'Image button'
(win_W, win_H) = (1024, 768)
sysPath = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(sysPath, "springs_04.MOV")

cap = cv2.VideoCapture()
states = {0: "Ready for Stream",
            1: "Streaming",
            2: "Recognizing"}
state = [states.get(0, "None")]

recognize = [False]

# YOLO
weights = os.path.join(sysPath, "lib/yolov3-custom.weights")
config = os.path.join(sysPath, "lib/yolov3-custom.cfg")
names = os.path.join(sysPath, "lib/custom.names")

yo = Yolo(weights, config, names)
yo.confidence = 0.2
yo.blobResize = 256

#
# UI Positioning 
#

# status position
stat_l_x, stat_l_y = int(0.02 * win_W), int(0.01 * win_H)
blink = 10
isBlink = [True]

# capture frame position
cap_fr1_x, cap_fr1_y, cap_fr1_w, cap_fr1_h = int(0.016 * win_W), int(0.04 * win_H), int(0.48 * win_W), int(0.54 * win_H)
cap_fr2_x, cap_fr2_y, cap_fr2_w, cap_fr2_h = int(0.51 * win_W), int(0.04 * win_H), int(0.48 * win_W), int(0.54 * win_H)

# action buttons
(start_b_x,
  start_b_y,
  start_b_w,
  start_b_h) = (int(0.016 * win_W), int(0.66 * win_H), int(0.113 * win_W), int(0.113 * win_W))
(recog_b_x,
  recog_b_y,
  recog_b_w,
  recog_b_h) = (int(0.15 * win_W), int(0.66 * win_H), int(0.14 * win_W), int(0.113 * win_W))
(reset_b_x,
  reset_b_y,
  reset_b_w,
  reset_b_h) = (int(0.07 * win_W), int(0.86 * win_H), int(0.14 * win_W), int(0.04 * win_H))
(stop_all_b_x,
  stop_all_b_y,
  stop_all_b_w,
  stop_all_b_h) = (int(0.314 * win_W), int(0.7 * win_H), int(0.2 * win_W), int(0.25 * win_H))

# trackbars
(trb_X_x, trb_X_y, trb_X_w) = (int(0.078 * win_W), int(0.586 * win_H), int(0.195 * win_W))
(trb_Y_x, trb_Y_y, trb_Y_w) = (int(0.275 * win_W), int(0.586 * win_H), int(0.195 * win_W))
(trb_W_x, trb_W_y, trb_W_w) = (int(0.472 * win_W), int(0.586 * win_H), int(0.195 * win_W))
(trb_H_x, trb_H_y, trb_H_w) = (int(0.669 * win_W), int(0.586 * win_H), int(0.195 * win_W))

# recognition region
(recog_res_x,
  recog_res_y,
  recog_res_w,
  recog_res_h) = (int(0.54 * win_W), int(0.66 * win_H), int(0.45 * win_W), int(0.33 * win_H))

# other
(max_width, max_height) = (int(0.48 * win_W), int(0.54 * win_H))

tr_vX = [0]
tr_vY = [0]
tr_vW = [cap_fr1_w]
tr_vH = [cap_fr1_h]

#
# Helpers
#
def setup():
    tr_vX = [0]
    tr_vY = [0]
    tr_vW = [cap_fr1_w]
    tr_vH = [cap_fr1_h]


def scaleImageToMax(image, max_w, max_h):
    w = image.shape[1]
    h = image.shape[0]

    k_w = max_w / w
    k_max_h = h * k_w

    if k_max_h > max_h:
        k_h = max_h / h
        return cv2.resize(image, (int(k_h * w), int(k_h * h)))
    else:
        return cv2.resize(image, (int(k_w * w), int(k_w * h)))

def detectFrom(img):
    yo.detectFrom(img)
    objects = yo.objects
    # print("[INFO] YOLO objects: {:}".format(len(objects)))

    # Visualize detected on source image
    font = cv2.FONT_HERSHEY_PLAIN
    for obj in objects:
        label = (
            obj.name
            + " {:}".format(objects.index(obj))
        )
        textColor = (0, 0, 0)
        boxColor = (150, 180, 20)
        cv2.rectangle(
            img,
            (obj.x, obj.y),
            (obj.x + obj.width, obj.y + obj.height),
            boxColor,
            1,
        )
        cv2.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)
    return objects

#
# Button actions
#
def startStream():
    cap.release()
    if not cap.isOpened():
        state[0] = states.get(1,"None")
        cap.open(path)#("rtsp://admin:@192.168.1.234:554")#
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def stopStream():
    state[0] = states.get(0,"None")
    recognize[0] = False
    cap.release()

def resetCrop():
    setup()

#### TODO
def recogCroppedImage():
    state[0] = states.get(2,"None")
    recognize[0]= True


#### TODO
def stopAll():
    stopStream()


#
# Main
#
def main():
    frame = np.zeros((win_H, win_W, 3), np.uint8)
    cvui.init(WINDOW_NAME)
    fr = 0
    while True:
        frame[:] = (0,0,0)

        objects = []

        if cap.isOpened():
            #main stream image
            ret, cv_frame_orig = cap.read()

            if ret:
                # status message blink
                fr += 1
                if fr % blink == 0:
                    isBlink[0] = not isBlink[0]
                if isBlink[0]:
                    cvui.text(frame, stat_l_x, stat_l_y, state[0])

                # original stream image
                cv_frame = scaleImageToMax(cv_frame_orig, max_width, max_height)
                cv_sh_h, cv_sh_w, _ = cv_frame.shape
                cv_sh_xx = cap_fr1_x + int((max_width - cv_sh_w) / 2)
                cv_sh_yy = cap_fr1_y + int((max_height - cv_sh_h) / 2)
                cvui.image(frame,  cv_sh_xx, cv_sh_yy, cv_frame)
                cv_frame_h, cv_frame_w, _ = cv_frame.shape

                #cropped stream image
                crop_image = cv_frame[int(tr_vY[0]):(int(tr_vH[0]) + int(tr_vY[0])), int(tr_vX[0]):(int(tr_vW[0]) + int(tr_vX[0]))]
                crop_image = scaleImageToMax(crop_image, max_width, max_height)
                cr_sh_h, cr_sh_w, _ = crop_image.shape
                cr_sh_xx = cap_fr2_x + int((max_width - cr_sh_w) / 2)
                cr_sh_yy = cap_fr2_y + int((max_height - cr_sh_h) / 2)

                if recognize[0] == True:
                    objects = detectFrom(crop_image)

                cvui.image(frame, cr_sh_xx, cr_sh_yy, crop_image)

                #track bars for cropping
                #x
                cvui.text(frame, trb_X_x + int(0.1 * win_W), trb_X_y - int(0.002 * win_H), "X", 0.5)
                if cvui.trackbar(frame, trb_X_x, trb_X_y, trb_X_w, tr_vX, 0., cv_sh_w):
                    cr_x = cv_sh_w - int(tr_vX[0])
                    if cr_x < tr_vW[0]:
                        tr_vW[0] = cr_x
                #y
                cvui.text(frame, trb_Y_x + int(0.1 * win_W), trb_Y_y - int(0.002 * win_H), "Y", 0.5)
                if cvui.trackbar(frame, trb_Y_x, trb_Y_y, trb_Y_w, tr_vY, 0., cv_sh_h):
                    cr_y = cv_sh_h - int(tr_vY[0])
                    if cr_y < tr_vH[0]:
                        tr_vH[0] = cr_y
                # w
                cvui.text(frame, trb_W_x + int(0.08 * win_W), trb_W_y - int(0.002 * win_H), "Width", 0.5)
                cvui.trackbar(frame, trb_W_x, trb_W_y, trb_W_w, tr_vW, tr_vX[0], cv_sh_w - int(tr_vX[0]))
                # h
                cvui.text(frame, trb_H_x + int(0.08 * win_W), trb_H_y - int(0.002 * win_H), "Height", 0.5)
                cvui.trackbar(frame, trb_H_x, trb_H_y, trb_H_w, tr_vH, tr_vY[0], cv_sh_h - int(tr_vY[0]))
                
                # zooming rect
                cvui.rect(frame, cv_sh_xx + int(tr_vX[0]), cv_sh_yy + int(tr_vY[0]), min(cv_sh_w, int(tr_vW[0])), min(cv_sh_h, int(tr_vH[0])), 0x00ff00)
            else:
                cap.release()
        else:
            cvui.text(frame, stat_l_x, stat_l_y, states.get(0, ""))
            
        # frame around image
        cvui.rect(frame, cap_fr1_x, cap_fr1_y, cap_fr1_w, cap_fr1_h, 0xffffff)
        cvui.rect(frame, cap_fr2_x, cap_fr2_y, cap_fr2_w, cap_fr2_h, 0xffffff)


        #start stream button
        if cvui.button(frame, start_b_x, start_b_y, start_b_w, start_b_h, "Start stream"):
            startStream()

        #reset all button
        # if cvui.button(frame, reset_b_x, reset_b_y, reset_b_w, reset_b_h, "Reset"):
        #     stopStream()

        #start recognition button
        if cvui.button(frame, recog_b_x, recog_b_y, recog_b_w, recog_b_h, "Start recognition"):
            recogCroppedImage()

        if cvui.button(frame, stop_all_b_x, stop_all_b_y, stop_all_b_w, stop_all_b_h, "STOP!"):
            stopAll()

        # recognition result region
        cvui.rect(frame, recog_res_x, recog_res_y, recog_res_w, recog_res_h, 0xffff00)

        obj_index = 0
        for obj in objects:
            label = (
            obj.name
            + " {:}: ".format(objects.index(obj))
            + "{:.6f}   ".format(obj.confidence)
            + "Center  x: "
            + str((obj.x + obj.width) / 2)
            + ", y: "
            + str((obj.y + obj.height) / 2)
        )  
            cvui.text(frame, recog_res_x + 20, recog_res_y + obj_index*20 + 20, label)
            obj_index += 1

        cvui.update()

        # Show everything on the screen
        cv2.imshow(WINDOW_NAME, frame)

        # Check if ESC key was pressed
        if cv2.waitKey(20) == 27:
            stopStream()
            break


if __name__ == '__main__':
    main()