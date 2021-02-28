import cv2
import numpy as np
import utils_image as u

positions = {0: "horizontal", 1: "vertical"}

class SlicedContour(object):
    def __init__(self, image, contour):
        self.rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(self.rect)
        self.box = np.int0(box)

        self.warped = u.crop_perpectively_min_area(image,contour)

        self.width = self.warped.shape[1]
        self.height = self.warped.shape[0]

        (h_l, h_r, h_t, h_b) = self.image_halfes(self.warped)

        self.halfs = [h_l, h_r, h_t, h_b]


        cnt_left = np.sum(h_l == 0)
        cnt_right = np.sum(h_r == 0)
        cnt_top = np.sum(h_t == 0)
        cnt_btm = np.sum(h_b == 0)

        self.half_counts = [cnt_left, cnt_right, cnt_top, cnt_btm]

        print("black pixels: {}, {}, {}, {}".format(cnt_left, cnt_right, cnt_top, cnt_btm))

        pos = positions.get(1)
        if self.width > self.height:
            pos = positions.get(0)

        self.position = pos

    def image_halfes(self,image):
        height = image.shape[0]
        width = image.shape[1]
        half_left = image[0:height, 0:int(width/2)]
        half_right = image[0:height, int(width/2):width]
        half_top = image[0:int(height/2), 0:width]
        half_btm = image[int(height/2):height, 0:width]
        return (half_left, half_right, half_top, half_btm)
