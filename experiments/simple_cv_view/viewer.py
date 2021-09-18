import cv2

cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.3:554/onvif1')

while True:
    res, img = cap.read()
    if img is not None:
        cv2.imshow("Frame", img)
        
    if cv2.waitKey(20) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()