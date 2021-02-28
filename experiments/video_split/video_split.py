import cv2 as cv
import os, sys

import argparse

# Defaults
scale = 0.5


# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="video source to split. Should be placed near, in \"sources\" folder.")
parser.add_argument("-s", "--scale", help="output images scale. Default = 0.5")

# Read arguments from the command line
args = parser.parse_args()
if args.input:
    sourceFile = args.input
else:
    parser.error("No video source file specified.")
    sys.exit(0)
if args.scale:
    scale = args.scale

(videoName, videoExt) = sourceFile.rsplit(".")

sysPath = os.path.dirname(os.path.abspath(__file__))

source = os.path.join(os.path.join(sysPath, "sources"), ".".join([videoName,videoExt]))
if not os.path.isfile(source):
    print("Input file ", source, " doesn't exist")
    sys.exit(1)


def folderExist(filePath):
    if not os.path.isdir(filePath):
        print("Folder ", filePath, " doesn't exist")
        return False
    else:
        return True


cap = cv.VideoCapture()
try:
    cap.open(source)
except:
    sys.exit(1)


i = 1
while True:
    ret, img = cap.read()

    if img is None:
        break

    resultFolder = os.path.join(sysPath, "result")
    if not folderExist(resultFolder):
        os.mkdir(resultFolder)
    resultFolder = os.path.join(resultFolder, videoName)
    if not folderExist(resultFolder):
        os.mkdir(resultFolder)

    height, width, _ = img.shape
    im_res = cv.resize(img, (int(width * scale), int(height * scale)))
    img = im_res

    cv.imwrite(os.path.join(resultFolder, "{:04d}".format(i) + ".jpg"), img)

    i += 1

cap.release()
