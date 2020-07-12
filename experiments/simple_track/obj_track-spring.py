import cv2 as cv
import os

source = "springs_11.mov"

sysPath = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(sysPath, source)

cap = cv.VideoCapture()
cap.open(source_path)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    frame = cv.resize(frame, (480, 960))
    ##transform
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, threshold = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    threshold = cv.medianBlur(threshold, 5)
    threshold = cv.medianBlur(threshold, 5)

    contour_frame = frame

    # Detect
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        print("area: {:}".format(area))
        # Distinguish small and big

        if 2000 < area < 6000:
            cv.rectangle(contour_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(contour_frame, str(area), (x, y), 1, 1, (0, 255, 0))

    ## present
    cv.imshow("stream", contour_frame)

    ch = cv.waitKey(1)
    if ch == 27:
        break

cap.release()
cv.destroyAllWindows
