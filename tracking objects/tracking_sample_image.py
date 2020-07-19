import cv2
import numpy as np

image = cv2.imread('sample.jpeg')

rx = 850
ry = 850
rw = 2800
rh = 2300
region = image[ry:rh, rx:rw]

gray_reg = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_reg, 180, 250, cv2.THRESH_BINARY)

(contours, _) = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if int(area) > 20000:
        rect = cv2.minAreaRect(cnt)
        (x,y),(w,h),angle = rect
        rect = (x+rx,y+ry),(w,h),angle

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(f"box: {box}")
        
        cv2.drawContours(image,[box],0,(0,0,255),3)
        cv2.putText(image,f"{int(x)}, {int(y)}, {int(w)}, {int(h)}, {int(angle)}", (int(x+rx),int(y+ry)), 4, 1, (0,0,254),2)

cv2.imshow("Frame", image)

while True:
    key = cv2.waitKey(1);
    if key == 27:
        break

cv2.destroyAllWindows