import cv2
import numpy as np

def crop_perpectively_min_area(img, cnt, draw_contour=False, debug=False):
    ## If contour box side length b[0]-b[1] greater than b[0]-b[3] - image will be 'vertical'
    ## If contour box side length b[0]-b[1] smaller than b[0]-b[3] - image will be 'horizontal'
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    if debug: print("bounding box: {}".format(box))
    if draw_contour:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    return cv2.warpPerspective(img, M, (width, height))

def medianPoint(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    return np.array([int(x),int(y)])

def draw_min_area_rect(image, contour, color=(255, 0, 0)):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, 1)