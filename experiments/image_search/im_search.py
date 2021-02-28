import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt
import os

sysPath = os.path.dirname(os.path.abspath(__file__))
base_image = "0081.jpg"
base_image_path = os.path.join(sysPath, base_image)#os.path.join(source_folder,source_file_name))

template_image = "1605_s2.png"
template_image_path = os.path.join(sysPath, template_image)#os.path.join(source_folder,source_file_name))

## --- Functions

def findTemplate(image_base, image_template):

    img_rgb = image_base##cv.imread(image_base)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(image_template,0)

    # template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where( res >= threshold)

    print("template loc: {}".format(loc))

    img = img_rgb

    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    return img
    # cv.imshow("sdfsd",img_rgb)


## --- Video

video_name = 'Skype_Video.mp4'

source = os.path.join(sysPath, video_name)
if not os.path.isfile(source):
    print("Input file ", source, " doesn't exist")
    sys.exit(1)

cap = cv.VideoCapture()
cap.open(source)

## check if we succeeded
if not cap.isOpened:
    sys.exit(1)

i = 1
while True:
    ret, img = cap.read()
    if not ret:
        break

    img_res = findTemplate(img, template_image_path)

    cv.imshow("Video", img_res)

    ch = cv.waitKey(1)
    if ch == 27:
        break
    i += 1


cap.release()
cv.destroyAllWindows()