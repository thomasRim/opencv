import cv2 as cv
import os
import sys
import utils
import yolo
import time

should_show_preview = True
should_write_result = False
detect_each_n_frame = 1
scale = 0.6

videoName = "springs_05"
videoExt = ".mov"
# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

weights = os.path.join(sysPath, "lib/yolov3-custom_sm.weights")
config = os.path.join(sysPath, "lib/yolov3-custom.cfg")
names = os.path.join(sysPath, "lib/custom.names")

yo = yolo.Yolo(weights, config, names)
yo.confidence = 0.15
yo.blobResize = 256

# Video
source = os.path.join(os.path.join(sysPath, "sources"), videoName + videoExt)
if not os.path.isfile(source):
    print("Input file ", source, " doesn't exist")
    sys.exit(1)

cap = cv.VideoCapture()
cap.open(source)

# check if we succeeded
if not cap.isOpened:
    sys.exit(1)

ret, img = cap.read()
height, width, _ = img.shape

capWrite = None
if should_write_result:
    resultFolder = os.path.join(sysPath, "result")
    if not utils.folderExist(resultFolder):
        os.mkdir(resultFolder)
    outSource = os.path.join(resultFolder, videoName + ".mp4")
    capWrite = cv.VideoWriter(
        outSource,
        0x7634706D,
        int(cap.get(cv.CAP_PROP_FPS)),
        (int(width * scale), int(height * scale)),
    )  # 0x7634706d - for mp4

i = 1
while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape
    img = cv.resize(img, (int(width * scale), int(height * scale)))

    if i % detect_each_n_frame == 0:  # each N frame, to fastener
        i = 0

        # show timing information on YOLO
        start = time.time()
        yo.detectFrom(img)
        end = time.time()
        print("[INFO] YOLO took {:.2f} seconds".format(end - start))

        # Visualize detected on source image
        font = cv.FONT_HERSHEY_PLAIN
        for obj in yo.objects:
            label = (
                obj.name
                + ": "
                + "{:.6f}".format(obj.confidence)
                + ": "
                + str(obj.x)
                + ", "
                + str(obj.y)
            )
            textColor = (0, 0, 0)
            boxColor = (150, 180, 20)
            cv.rectangle(
                img,
                (obj.x, obj.y),
                (obj.x + obj.width, obj.y + obj.height),
                boxColor,
                1,
            )
            cv.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)

    if should_show_preview:
        cv.imshow("Video", img)

    if capWrite and should_write_result:
        capWrite.write(img)

    ch = cv.waitKey(1)
    if ch == 27:
        break
    i += 1


if capWrite:
    capWrite.release()
cap.release()
cv.destroyAllWindows()
