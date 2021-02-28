import numpy as np
import cv2
import os, sys
# from matplotlib import pyplot as plt

filename = "saved_pypylon_img.png"#"pypylon_img.png"#"shapes.jpg"
sysPath = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(sysPath, filename)

img = cv2.imread(source,0)
edges = cv2.Canny(img,200,230)

cv2.imshow("",edges)


# img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold", threshold)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
print("contours: ", len(contours))
font = cv2.FONT_HERSHEY_COMPLEX

unique = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.001*cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    print(area," ", x, " ", y)


    # if unique.__contains__((cnt.x))


    if int(area) > 10000 and int(area) < 70000:

        cv2.drawContours(img, [approx], 0, (0), 1)
        

        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 0.5, (0))
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), font, 0.5, (0))
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 0.5, (0))
        else:
            cv2.putText(img, "Circle", (x, y), font, 0.5, (0))
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img,center,radius,(0,255,0),2)

cv2.imshow("shapes", img)

cv2.waitKey(0)
cv2.destroyAllWindows()