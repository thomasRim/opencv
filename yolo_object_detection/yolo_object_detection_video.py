import cv2 as cv
import os
import sys
import utils
import yolo
import time

# Net
yo = yolo.Yolo()

# Video
sysPath = os.path.dirname(os.path.abspath(__file__))

videoName = 'vtest'
source = os.path.join(os.path.join(sysPath, 'sources'),videoName+'.avi')
if not os.path.isfile(source):
        print("Input file ", source, " doesn't exist")
        sys.exit(1)

# source = 'http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8?dummy=param.mjpg'
# source = 'http://devimages.apple.com/iphone/samples/bipbop/gear1/prog_index.m3u8?dummy=param.mjpg'
# source = 'https://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8'
# source = 'https://bitmovin-a.akamaihd.net/content/playhouse-vr/m3u8s/105560.m3u8'
# source = 'https://bitdash-a.akamaihd.net/content/MI201109210084_1/m3u8s/f08e80da-bf1d-4e3d-8899-f0f6155f6efa.m3u8'
cap = cv.VideoCapture()
cap.open(source)

#check if we succeeded
if not cap.isOpened:
    sys.exit(1)

capWrite = None

ret, img = cap.read()
# if ret:
#     height, width, channels = img.shape
#     resultFolder = os.path.join(sysPath, 'result')
#     if not utils.folderExist(resultFolder):
#         os.mkdir(resultFolder)
#     outSource = os.path.join(resultFolder,videoName)
#     capWrite = cv.VideoWriter(outSource, 0x7634706d, int(cap.get(cv.CAP_PROP_FPS)), (width, height) ) #0x7634706d - for mp4

i = 1
while True:
    ret, img = cap.read()
    if not ret:
        break
    if i % 5 == 0 : # each 5 frame, to fastener
        i = 0

        # show timing information on YOLO
        start = time.time()
        yo.detectFrom(img) 
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        cv.imshow('Video', img)

    if capWrite:
        capWrite.write(img)
    ch = cv.waitKey(1)
    if ch == 27:
        break
    i += 1

if capWrite:
    capWrite.release()
cap.release()
cv.destroyAllWindows()