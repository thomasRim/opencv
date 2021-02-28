from pypylon import pylon
import cv2
import time

cap_size = (1278, 958)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
filename = "output_{:}.avi".format(time.strftime("%H-%M-%S"))
video_writer = cv2.VideoWriter(filename, fourcc, 30, cap_size)

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

frame = 0
max_frames = 30 * 60 * 5
while camera.IsGrabbing():
    if frame >= max_frames:
        break

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        video_writer.write(img)
        frame += 1
    # grabResult.Release()

camera.StopGrabbing()
video_writer.release()
camera.Close()