#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

from typing import Any

import numpy as np
import cv2 as cv
import os
from multiprocessing.pool import ThreadPool
from collections import deque

from  common import clock, draw_str, StatValue
import video
from yolo import Yolo


# class DummyTask:
#     def __init__(self, data):
#         self.data = data
#     def ready(self):
#         return True
#     def get(self):
#         return self.data

def main():
    import sys

    # try:
    #     fn = sys.argv[1]
    # except:
    #     fn = 0
    # cap = video.create_capture(fn)

    def fileExist(filePath):
        if not os.path.isfile(filePath):
            print("Input file ", filePath, " doesn't exist")
            sys.exit(1)
    sysPath = os.path.dirname(os.path.abspath(__file__))

    # Load Yolo
    weightPath = os.path.join(sysPath, 'lib/yolov3-custom.weights')
    configPath = os.path.join(sysPath, 'lib/yolov3-custom.cfg')
    fileExist(weightPath)
    fileExist(configPath)

    # Net
    net = cv.dnn.readNet(weightPath, configPath)
    # net = cv.dnn.readNetFromDarknet(configPath, weightPath)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

    # classes
    classes = []
    cocoNamesPath = os.path.join(sysPath, 'lib/custom.names')
    fileExist(cocoNamesPath)
    with open(cocoNamesPath, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Video
    videoName = 'springs_05.MOV'
    source = os.path.join(os.path.join(sysPath, 'sources'), videoName)
    fileExist(source)
    cap = cv.VideoCapture(source)

    yo = Yolo(weightPath, configPath, cocoNamesPath)

    def process_frame(frame, t0):
        objects = yo.detectFrom(frame)
        print("detected:" + str(objects))
        return frame, t0

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value*1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv.imshow('threaded video', res)
        if len(pending) < threadn:
            _ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
            pending.append(task)

        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()