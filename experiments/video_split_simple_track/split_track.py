import cv2 as cv
import os, time
import numpy as np

#### Helpers
def folderExist(filePath):
    if not os.path.isdir(filePath):
        print("Folder ", filePath, " doesn't exist")
        return False
    else:
        return True


def gammaCorrection(image):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(image, lookUpTable)
    return res


#### Setup
# source
source = "springs_11"  # 04, 06, 08, 09, ~10
source_ext = ".mov"
classNames = ["spring"]

# options
cases = {
    "split_only": (True, False, False, False),
    "split_detect": (True, True, False, False),
    "detect_preview": (False, True, True, False),
    "detect_preview_raw": (False, True, True, True),
}

case = cases.get("split_detect", (False, False, False, False))
print(case)
(split_images, find_objects, preview_objects, preview_treshold) = case

scale = 1
each_n_frame = 10
frame_delay = 0.2

gamma = 1
alpha = 1.35  # Simple contrast control [1.0-3.0]
beta = 5  # Simple brightness control [0-100]

#### Open source
#
sysPath = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(sysPath, source + source_ext)

cap = cv.VideoCapture()
try:
    cap.open(source_path)
except:
    exit(0)

#### Loop
#
i = 1
n = 0
while True:
    ret, img = cap.read()

    if img is None:
        print("no image source")
        break
    n += 1

    if n % each_n_frame == 0:
        height, width, _ = img.shape
        img = cv.resize(img, (int(width * scale), int(height * scale)))

        sc_height, sc_width, _ = img.shape
        ## save each image to folder equal to source name
        if split_images:
            resultFolder = os.path.join(sysPath, "result")
            if not folderExist(resultFolder):
                os.mkdir(resultFolder)
            resultFolder = os.path.join(resultFolder, source)
            if not folderExist(resultFolder):
                os.mkdir(resultFolder)
            name = "{:04d}".format(i) + ".jpg"
            print(name)
            cv.imwrite(os.path.join(resultFolder, name), img)

        ## filter image to find stale objects
        if find_objects:
            img = gammaCorrection(img)
            img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, threshold = cv.threshold(gray, 145, 255, cv.THRESH_BINARY)
            threshold = cv.medianBlur(threshold, 5)
            threshold = cv.medianBlur(threshold, 5)

            # Detect
            contours, _ = cv.findContours(
                threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
            )
            cntrs_cnt = 0
            for cnt in contours:
                (x, y, w, h) = cv.boundingRect(cnt)
                area = cv.contourArea(cnt)
                print("area: {:}".format(area))
                # Distinguish small and big

                if 1400 < area < 10000:
                    cntrs_cnt += 1
                    if preview_objects:
                        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv.putText(img, str(area), (x, y), 1, 1, (0, 255, 0))
                    if split_images:
                        file = open(
                            os.path.join(resultFolder, "{:04d}".format(i) + ".txt"), "a"
                        )
                        file.writelines(
                            "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                                (x + w / 2) / sc_width,
                                float(y + h / 2) / sc_height,
                                w / sc_width,
                                h / sc_height,
                            )
                        )
            print("detect contours: {:}".format(cntrs_cnt))

            if preview_objects:
                preview_image = img
                if preview_treshold:
                    preview_image = threshold
                cv.imshow("stream", preview_image)

                ch = cv.waitKey(1)
                if ch == 27:
                    break

                time.sleep(frame_delay)

        i += 1

if find_objects:
    file = open(os.path.join(resultFolder, "classes.txt"), "w")
    for name in classNames:
        file.write(name + "\n")

cap.release()
cv.destroyAllWindows
