import numpy as np
import cv2
import cvui
import os

from yolo import Yolo

WINDOW_NAME = "Image button"
(win_W, win_H) = (1024, 768)
sysPath = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(sysPath, "output_21-19-51.avi")

cap = cv2.VideoCapture()

# YOLO
weights = os.path.join(sysPath, "lib/yolov3-custom.weights")
config = os.path.join(sysPath, "lib/yolov3-custom.cfg")
names = os.path.join(sysPath, "lib/custom.names")

yo = Yolo(weights, config, names)
yo.confidence = 0.2
yo.blobResize = 256

# cap/video state
states = {0: "Ready for Stream", 1: "Streaming", 2: "Recognizing"}
state = [states.get(0, "None")]

# recognition button states
recog_btn_states = {0: "Start recognition", 1: "Stop recognition"}
recog_btn_state = [recog_btn_states.get(0, "None")]

# recognizers state
recog_states = {0: "Contours >", 1: "< YOLO"}
recog_state = [recog_states.get(0, "None")]

# contour debug state
contour_states = {0: "normal", 1: "thresholded"}
contour_state = [contour_states.get(0, "")]

blink = 10
isBlink = [True]
recognize = [False]

# 0 - Simple contrast control [1.0-3.0]; 1 - Simple brightness control [0-100]; 2 - gamma
img_alpha = [1.35]
img_beta = [5]
img_gamma = [1]
# threshold space min/max 0-255
threshold_min = [140]
threshold_max = [255]
# contour area min/max in pixel
contour_area_min = [1400]
contour_area_max = [10000]


#
# UI Positioning
#

# status position
stat_l_x, stat_l_y = int(0.02 * win_W), int(0.01 * win_H)

# capture frame position
cap_fr1_x, cap_fr1_y, cap_fr1_w, cap_fr1_h = (
    int(0.016 * win_W),
    int(0.04 * win_H),
    int(0.48 * win_W),
    int(0.54 * win_H),
)
cap_fr2_x, cap_fr2_y, cap_fr2_w, cap_fr2_h = (
    int(0.51 * win_W),
    int(0.04 * win_H),
    int(0.48 * win_W),
    int(0.54 * win_H),
)

# action buttons
(start_b_x, start_b_y, start_b_w, start_b_h) = (
    int(0.016 * win_W),
    int(0.66 * win_H),
    int(0.113 * win_W),
    int(0.113 * win_W),
)
(recog_b_x, recog_b_y, recog_b_w, recog_b_h) = (
    int(0.15 * win_W),
    int(0.66 * win_H),
    int(0.14 * win_W),
    int(0.113 * win_W),
)
(stop_all_b_x, stop_all_b_y, stop_all_b_w, stop_all_b_h) = (
    int(0.016 * win_W),
    int(0.83 * win_H),
    int(0.274 * win_W),
    int(0.16 * win_H),
)
(reset_crop_b_x, reset_crop_b_y, reset_crop_b_w, reset_crop_b_h) = (
    int(0.016 * win_W),
    int(0.586 * win_H),
    int(0.06 * win_W),
    int(0.06 * win_H),
)


# crop trackbars
## frame_x
(trb_X_x, trb_X_y, trb_X_w) = (
    int(0.078 * win_W),
    int(0.586 * win_H),
    int(0.195 * win_W),
)
## frame y
(trb_Y_x, trb_Y_y, trb_Y_w) = (
    int(0.275 * win_W),
    int(0.586 * win_H),
    int(0.195 * win_W),
)
## frame width
(trb_W_x, trb_W_y, trb_W_w) = (
    int(0.472 * win_W),
    int(0.586 * win_H),
    int(0.195 * win_W),
)
## frame height
(trb_H_x, trb_H_y, trb_H_w) = (
    int(0.669 * win_W),
    int(0.586 * win_H),
    int(0.195 * win_W),
)

# recognizers setup region
(recog_set_x, recog_set_y, recog_set_w, recog_set_h) = (
    int(0.3 * win_W),
    int(0.66 * win_H),
    int(0.33 * win_W),
    int(0.33 * win_H),
)

# recognizers setup toggle button - yolo/contour
(recog_set_btn_x, recog_set_btn_y, recog_set_btn_w, recog_set_btn_h) = (
    int(0.305 * win_W),
    int(0.665 * win_H),
    int(0.32 * win_W),
    int(0.03 * win_H),
)

#resognizer contour normal/binary toggle button
(cnt_state_btn_x, cnt_state_btn_y, cnt_state_btn_w, cnt_state_btn_h) = (
    int(0.88 * win_W),
    int(0.586 * win_H),
    int(0.1 * win_W),
    int(0.03 * win_H),
)
# recognizers trackbars
## threshold min
(trecog_thr_min_x, trecog_thr_min_y, trecog_thr_min_w) = (
    int(0.42 * win_W),
    int(0.7 * win_H),
    int(0.195 * win_W),
)
## threshold max
(trecog_thr_max_x, trecog_thr_max_y, trecog_thr_max_w) = (
    int(0.42 * win_W),
    int(0.75 * win_H),
    int(0.195 * win_W),
)
## alpha
(trecog_alpha_x, trecog_alpha_y, trecog_alpha_w) = (
    int(0.42 * win_W),
    int(0.8 * win_H),
    int(0.195 * win_W),
)
## beta
(trecog_beta_x, trecog_beta_y, trecog_beta_w) = (
    int(0.42 * win_W),
    int(0.85 * win_H),
    int(0.195 * win_W),
)
## contour area min
(trecog_area_min_x, trecog_area_min_y, trecog_area_min_w) = (
    int(0.42 * win_W),
    int(0.9 * win_H),
    int(0.195 * win_W),
)
## contour area max
(trecog_area_max_x, trecog_area_max_y, trecog_area_max_w) = (
    int(0.42 * win_W),
    int(0.95 * win_H),
    int(0.195 * win_W),
)


# recognition result region
(recog_res_x, recog_res_y, recog_res_w, recog_res_h) = (
    int(0.64 * win_W),
    int(0.66 * win_H),
    int(0.35 * win_W),
    int(0.33 * win_H),
)

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
    tr_vX[0] = 0
    tr_vY[0] = 0
    tr_vW[0] = cap_fr1_w
    tr_vH[0] = cap_fr1_h


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

def gammaCorrection(image, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(image, lookUpTable)
    return res


def toggleRecognizers():
    if recog_state[0] == recog_states.get(1, ""):
        recog_state[0] = recog_states.get(0, "")
        ## some controls for YOLO
    else:
        recog_state[0] = recog_states.get(1, "")


# YOLO
def detectObjectsFrom(img, frame):
    yo.detectFrom(img)
    objects = yo.objects
    # print("[INFO] YOLO objects: {:}".format(len(objects)))

    # Visualize detected on source image
    font = cv2.FONT_HERSHEY_PLAIN
    for obj in objects:
        label = obj.name + " {:}".format(objects.index(obj))
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

    obj_index = 0
    for obj in objects:
        label = (
            obj.name
            + " {:}: ".format(objects.index(obj))
            + "{:.6f}   ".format(obj.confidence)
            + "; x: "
            + str((obj.x + obj.width) / 2)
            + ", y: "
            + str((obj.y + obj.height) / 2)
        )
        cvui.text(frame, recog_res_x + 20, recog_res_y + obj_index * 20 + 20, label)
        obj_index += 1


# Contours
def correctedImage(img, alpha, beta, gamma):
    corrected_img = gammaCorrection(img, gamma)
    converted_img = cv2.convertScaleAbs(corrected_img, alpha=alpha, beta=beta)

    gray = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(
        gray, threshold_min[0], threshold_max[0], cv2.THRESH_BINARY
    )
    threshold_img = cv2.medianBlur(threshold_img, 5)
    threshold_img = cv2.medianBlur(threshold_img, 5)
    return threshold_img

def detectContoursFrom(img, frame, alpha, beta, gamma, area_min, area_max):
    threshold_img = correctedImage(img, alpha, beta, gamma)

    # Detect
    contours, _ = cv2.findContours(
        threshold_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cntrs_cnt = 0
    good_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("area: {:}".format(area))
        # Distinguish small and big

        if area_min < area < area_max:
            good_contours.append(cnt)
            cntrs_cnt += 1

    print("detect contours: {:}".format(cntrs_cnt))

    obj_index = 0

    for cnt in good_contours:
        # draw minimum area box rotated
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  ##
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        ((rcx, rcy), (rw, rh), angle) = rect

        if rw < rh:
            angle = angle - 90

        print("bos: {:.4f}".format(angle))
        # draw bounding contour box
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img, "{:}, {}".format(obj_index, str(area)), (x, y), 1, 1, (0, 255, 0)
        )

        # setup and draw result label
        label = (
            "{:}: ".format(obj_index)
            + "; x: "
            + str((x + w) / 2)
            + ", y: "
            + str((y + h) / 2)
            + ", area: {:}".format(cv2.contourArea(cnt))
            + ", deg: {:.1f}".format(angle)
        )
        cvui.text(frame, recog_res_x + 10, recog_res_y + obj_index * 20 + 20, label)
        obj_index += 1


#
# Button actions
#

#### Stream
def startStream():
    cap.release()
    if not cap.isOpened():
        state[0] = states.get(1, "None")
        cap.open(path)  # ("rtsp://admin:@192.168.1.234:554")#
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def stopStream():
    state[0] = states.get(0, "None")
    recognize[0] = False
    cap.release()


def resetCrop():
    setup()


#### Recognition
def startRecognize():
    state[0] = states.get(2, "None")
    recog_btn_state[0] = recog_btn_states.get(1, "None")
    recognize[0] = True

def stopRecognize():
    state[0] = states.get(1, "None")
    recog_btn_state[0] = recog_btn_states.get(0, "None")
    recognize[0] = False

def contourStateToggle():
    if contour_state[0] == contour_states.get(0,""):
        contour_state[0] = contour_states.get(1,"")
    else:
        contour_state[0] = contour_states.get(0,"")
    setup()

#### TODO
def stopAll():
    stopStream()


def cropTrackBars(frame, cv_sh_w, cv_sh_h):
    # reset crop button
    if cvui.button(
        frame, reset_crop_b_x, reset_crop_b_y, reset_crop_b_w, reset_crop_b_h, "Reset"
    ):
        resetCrop()
    # x
    cvui.text(frame, trb_X_x + int(0.1 * win_W), trb_X_y - int(0.002 * win_H), "X", 0.5)
    if cvui.trackbar(frame, trb_X_x, trb_X_y, trb_X_w, tr_vX, 0.0, cv_sh_w):
        cr_x = cv_sh_w - int(tr_vX[0])
        if cr_x < tr_vW[0]:
            tr_vW[0] = cr_x
    # y
    cvui.text(frame, trb_Y_x + int(0.1 * win_W), trb_Y_y - int(0.002 * win_H), "Y", 0.5)
    if cvui.trackbar(frame, trb_Y_x, trb_Y_y, trb_Y_w, tr_vY, 0.0, cv_sh_h):
        cr_y = cv_sh_h - int(tr_vY[0])
        if cr_y < tr_vH[0]:
            tr_vH[0] = cr_y
    # w
    cvui.text(
        frame, trb_W_x + int(0.08 * win_W), trb_W_y - int(0.002 * win_H), "Width", 0.5
    )
    cvui.trackbar(
        frame, trb_W_x, trb_W_y, trb_W_w, tr_vW, tr_vX[0], cv_sh_w - int(tr_vX[0])
    )
    # h
    cvui.text(
        frame, trb_H_x + int(0.08 * win_W), trb_H_y - int(0.002 * win_H), "Height", 0.5
    )
    cvui.trackbar(
        frame, trb_H_x, trb_H_y, trb_H_w, tr_vH, tr_vY[0], cv_sh_h - int(tr_vY[0])
    )


def contourTrackBars(frame):
    # threshold
    cvui.text(
        frame,
        trecog_thr_min_x - int(0.1 * win_W),
        trecog_thr_min_y + int(0.015 * win_H),
        "Threshold min",
    )
    cvui.trackbar(
        frame,
        trecog_thr_min_x,
        trecog_thr_min_y,
        trecog_thr_min_w,
        threshold_min,
        1,
        threshold_max[0] - 1,
    )

    cvui.text(
        frame,
        trecog_thr_max_x - int(0.1 * win_W),
        trecog_thr_max_y + int(0.01 * win_H),
        "Threshold max",
    )
    cvui.trackbar(
        frame,
        trecog_thr_max_x,
        trecog_thr_max_y,
        trecog_thr_max_w,
        threshold_max,
        threshold_min[0],
        255,
    )

    # image correction
    cvui.text(
        frame,
        trecog_alpha_x - int(0.1 * win_W),
        trecog_alpha_y + int(0.015 * win_H),
        "Correct Alpha",
    )
    cvui.trackbar(
        frame, trecog_alpha_x, trecog_alpha_y, trecog_alpha_w, img_alpha, 1.0, 3.0, 0.1
    )

    cvui.text(
        frame,
        trecog_beta_x - int(0.1 * win_W),
        trecog_beta_y + int(0.015 * win_H),
        "Correct Beta",
    )
    cvui.trackbar(frame, trecog_beta_x, trecog_beta_y, trecog_beta_w, img_beta, 1, 100)

    # area
    cvui.text(
        frame,
        trecog_area_min_x - int(0.1 * win_W),
        trecog_area_min_y + int(0.015 * win_H),
        "Area min",
    )
    cvui.trackbar(
        frame,
        trecog_area_min_x,
        trecog_area_min_y,
        trecog_area_min_w,
        contour_area_min,
        1.0,
        contour_area_max[0] - 1,
    )

    cvui.text(
        frame,
        trecog_area_max_x - int(0.1 * win_W),
        trecog_area_max_y + int(0.015 * win_H),
        "Area max",
    )
    cvui.trackbar(
        frame,
        trecog_area_max_x,
        trecog_area_max_y,
        trecog_area_max_w,
        contour_area_max,
        contour_area_min[0],
        20000,
    )


def main():
    frame = np.zeros((win_H, win_W, 3), np.uint8)
    cvui.init(WINDOW_NAME)
    fr = 0
    while True:
        frame[:] = (0, 0, 0)

        objects = []

        if cap.isOpened():
            # main stream image
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
                cvui.image(frame, cv_sh_xx, cv_sh_yy, cv_frame)
                cv_frame_h, cv_frame_w, _ = cv_frame.shape

                # cropped stream image
                crop_image = cv_frame[
                    int(tr_vY[0]) : (int(tr_vH[0]) + int(tr_vY[0])),
                    int(tr_vX[0]) : (int(tr_vW[0]) + int(tr_vX[0])),
                ]
                crop__scaled_image = scaleImageToMax(crop_image, max_width, max_height)
                cr_sh_h, cr_sh_w, _ = crop__scaled_image.shape
                cr_sh_xx = cap_fr2_x + int((max_width - cr_sh_w) / 2)
                cr_sh_yy = cap_fr2_y + int((max_height - cr_sh_h) / 2)

                # crop track bars
                cropTrackBars(frame, cv_sh_w, cv_sh_h)

                # zooming rect
                cvui.rect(
                    frame,
                    cv_sh_xx + int(tr_vX[0]),
                    cv_sh_yy + int(tr_vY[0]),
                    min(cv_sh_w, int(tr_vW[0])),
                    min(cv_sh_h, int(tr_vH[0])),
                    0x00FF00,
                )

                ## Setup recognition UI
                # present toggle to switch 'YOLO' and 'filters'. Toggle will drop recognition - need manual start
                # 'filter' will have adjustable trackbars for threshold (2), alpha, beta, area_min, area_max = 6 ,
                if cvui.button(
                    frame,
                    recog_set_btn_x,
                    recog_set_btn_y,
                    recog_set_btn_w,
                    recog_set_btn_h,
                    recog_state[0],
                ):
                    stopRecognize()
                    toggleRecognizers()

                # recognition of objects and data presenting
                if recognize[0] == True:
                    if recog_state[0] == recog_states.get(0, ""):
                        detectContoursFrom(
                            crop__scaled_image,
                            frame,
                            img_alpha[0],
                            img_beta[0],
                            img_gamma[0],
                            contour_area_min[0],
                            contour_area_max[0],
                        )
                        contourTrackBars(frame)

                    else:
                        detectObjectsFrom(crop__scaled_image, frame)

                # draw result
                if contour_state[0] == contour_states.get(0,""):
                    cvui.image(frame, cr_sh_xx, cr_sh_yy, crop__scaled_image)
                else:
                    colored_grey = cv2.cvtColor(correctedImage(crop__scaled_image, img_alpha[0],
                                img_beta[0],
                                img_gamma[0]), cv2.COLOR_GRAY2BGR)
                    cvui.image(frame, cr_sh_xx, cr_sh_yy, colored_grey)

            else:
                cap.release()
        else:
            cvui.text(frame, stat_l_x, stat_l_y, states.get(0, ""))

        ##
        ## Static UI
        ##

        # frame around image
        cvui.rect(frame, cap_fr1_x, cap_fr1_y, cap_fr1_w, cap_fr1_h, 0xFFFFFF)
        cvui.rect(frame, cap_fr2_x, cap_fr2_y, cap_fr2_w, cap_fr2_h, 0xFFFFFF)

        # start stream button
        if cvui.button(
            frame, start_b_x, start_b_y, start_b_w, start_b_h, "Start stream"
        ):
            startStream()

        # start recognition button
        if cvui.button(
            frame, recog_b_x, recog_b_y, recog_b_w, recog_b_h, recog_btn_state[0]
        ):
            if recognize[0] == True:
                stopRecognize()
            else:
                startRecognize()

        # stop all button
        if cvui.button(
            frame, stop_all_b_x, stop_all_b_y, stop_all_b_w, stop_all_b_h, "STOP!"
        ):
            stopAll()

        # recognizers setup region
        cvui.rect(frame, recog_set_x, recog_set_y, recog_set_w, recog_set_h, 0x008EDF)

        # contour toggle
        if recognize[0] == True:
            if recog_state[0] == recog_states.get(0,""):
                if cvui.button(frame, cnt_state_btn_x, cnt_state_btn_y, cnt_state_btn_w, cnt_state_btn_h, contour_state[0]):
                    contourStateToggle()

        # recognition result region
        cvui.rect(frame, recog_res_x, recog_res_y, recog_res_w, recog_res_h, 0xFFFF00)

        # update all ui
        cvui.update()

        # Show everything on the screen
        cv2.imshow(WINDOW_NAME, frame)

        # Check if ESC key was pressed
        if cv2.waitKey(20) == 27:
            stopStream()
            break


if __name__ == "__main__":
    main()