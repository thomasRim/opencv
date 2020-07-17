import cv2
import webcamvideostream as ws

cap = ws.WebcamVideoStream(src="rtsp://admin:@192.168.1.234:554")
cap.start()

while True:
    frame = cap.read()

    # Show everything on the screen
    cv2.imshow("WINDOW_NAME", frame)

    # Check if ESC key was pressed
    if cv2.waitKey(20) == 27:
        break

cap.stop()
cv2.destroyAllWindows()