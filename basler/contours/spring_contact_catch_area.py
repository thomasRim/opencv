import cv2
import numpy as np
import os
import utils_image as u
import sliced_contour as sc

################################################################
# Setup

contour_min = 1000
contour_max = 10000
file_name = 'red.png'
debug = False

################################################################
# Functions
#

def find_image_dock(image, threshold, contour, index=0):
    quarters = {0: "lb", 1: "lt", 2: "rt", 3: "rb"}
    sliced_image = sc.SlicedContour(threshold, contour)

    ##
    ## define which quarter half has dock area
    ##

    q = quarters.get(0)

    counts = sliced_image.half_counts

    if sliced_image.position == sc.positions.get(0) : # hotizontal
        if counts[0] > counts[1]:
            if counts[2] > counts[3]:   q = quarters.get(2)
            else:                       q = quarters.get(3)
        else:
            if counts[2] > counts[3]:   q = quarters.get(1)
            else:                       q = quarters.get(0)
    else: # vertical
        if counts[0] > counts[1]:
            if counts[2] > counts[3]:   q = quarters.get(0)
            else:                       q = quarters.get(1)
        else:
            if counts[2] > counts[3]:   q = quarters.get(3)
            else:                       q = quarters.get(2)

    print(q)

    ##
    ## mark quarter' inner half as dock point
    ##

    rect = sliced_image.rect
    box = sliced_image.box
    width = sliced_image.width
    height = sliced_image.height

    test_box = np.array(box,copy=False)

    p01 = u.medianPoint(box[0], box[1])
    p12 = u.medianPoint(box[1], box[2])
    p23 = u.medianPoint(box[2], box[3])
    p03 = u.medianPoint(box[0], box[3])
    pc = np.array([int(rect[0][0]), int(rect[0][1])])

    if q == quarters.get(0): #left-btm
        if width > height:  test_box = np.array([ u.medianPoint(box[0], p03), u.medianPoint(p01, pc), pc, p03])
        else:               test_box = np.array([ u.medianPoint(box[0], p01), p01, pc, u.medianPoint(pc, p03) ])
    elif q == quarters.get(1): #left-top
        if width > height:  test_box = np.array([ u.medianPoint(p01, pc), u.medianPoint(box[1], p12), p12, pc ])
        else:               test_box = np.array([ p01, u.medianPoint(p01, box[1]), u.medianPoint(p12, pc), pc ])
    elif q == quarters.get(2): #right-top
        if width > height:  test_box = np.array([ pc, p12, u.medianPoint(p12, box[2]), u.medianPoint(pc, p23) ])
        else:               test_box = np.array([ pc, u.medianPoint(p12, pc), u.medianPoint(box[2], p23), p23 ])
    else: # right-btm
        if width > height:  test_box = np.array([ p03, pc, u.medianPoint(pc, p23), u.medianPoint(p03, box[3]) ])
        else:               test_box = np.array([ u.medianPoint(p03, pc), pc, p23, u.medianPoint(p23, box[3]) ])
        
    cv2.drawContours(image, [test_box], 0, (255, 0, 0), 1)

    if debug: cv2.putText(image, "{} {}:{}".format(index,sliced_image.position,q), (int(rect[0][0]), int(rect[0][1])),1,1, (0.255,0))


################################################################
# Sys

sysPath = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(sysPath, file_name)
image = cv2.imread(file_path)

################################################################
# Threshold 

gray_reg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_reg, 70, 250, cv2.THRESH_BINARY)

if debug: cv2.imshow("Threshold", threshold)

################################################################
# Contours

(contours, _) = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if contour_min < int(area) < contour_max:
        find_image_dock(image, threshold, cnt, i)

        # u.draw_min_area_rect(image, cnt)

        
        # ##
        # ## Try find defective
        # ##

        sliced_image = sc.SlicedContour(threshold, cnt)
        halfs = sliced_image.halfs
        counts = sliced_image.half_counts

        defect_try_image = np.array([])

        if sliced_image.position == sc.positions.get(0) : # horizontal
            if counts[0] > counts[1] :  defect_try_image = halfs[0]
            else :                      defect_try_image = halfs[1]
        else : # vertical
            if counts[2] > counts[3] :  defect_try_image = halfs[2]
            else :                      defect_try_image = halfs[3]


        x_offset=y_offset=10

        dty_h, dty_w = defect_try_image.shape

        larger = np.zeros([dty_h+y_offset*2, dty_w+x_offset*2],dtype=np.uint8)
        larger.fill(255)

        larger[y_offset:y_offset+dty_h, x_offset:x_offset+dty_w] = defect_try_image

        larger =  cv2.medianBlur(larger,3)

        cv2.imshow("{}".format(i),larger)


        defect_try_image = larger

        (def_contours,_) = cv2.findContours(defect_try_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        h,w = defect_try_image.shape

        def_half_size = (h * w) / 2

        print("half area: {}".format(def_half_size))

        for d_i, def_cnt in enumerate(def_contours):
            def_area = cv2.contourArea(def_cnt)
            if debug: print("{}:{}: def sub cont area: {}".format(i,d_i,def_area))
            if  def_half_size > def_area >= 1000 :
                # u.draw_min_area_rect(defect_try_image, def_cnt)
                def_warped = u.crop_perpectively_min_area(defect_try_image,def_cnt, debug)
                dh,dw = def_warped.shape
                r_k = min(dw,dh)/max(dw,dh)
                # cv2.imshow("{}:dfw: {}, {}".format(i,d_i, r_k),def_warped)

                if r_k < 0.6 :
                    u.draw_min_area_rect(image, cnt, (255,255,0))


        

################################################################
# Present

cv2.imshow("Frame", image)

while True:
    key = cv2.waitKey(1);
    if key == 27:
        break

cv2.destroyAllWindows