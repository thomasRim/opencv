from darknetpy.detector import Detector
from matplotlib import image, patches, pyplot as plt
import cv2, os

scale = 0.4

detector = Detector("custom.data", "yolov3-custom.cfg", "yolov3-custom.weights",)

sysPath = os.path.dirname(os.path.abspath(__file__))
imagePath = os.path.join(sysPath, "spring.jpeg")

boxes = detector.detect("spring.jpeg")
print(boxes)

img = cv2.imread(imagePath)

h, w, _ = img.shape
print("w:{:}, h:{:}".format(w, h))

for i, box in enumerate(boxes):
    c = box["class"]
    p = box["prob"]
    l = box["left"]
    r = box["right"]
    t = box["top"]
    b = box["bottom"]

    cv2.rectangle(
        img, (int(l), int(t)), (int(r - l), int(b - t)), (0, 255, 0), 1,
    )

img = cv2.resize(img, (int(scale * w), int(scale * h)))
cv2.imshow("Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
