from pypylon import pylon
import time

num_img_to_save = 5
img = pylon.PylonImage()
tlf = pylon.TlFactory.GetInstance()

cam = pylon.InstantCamera(tlf.CreateFirstDevice())
cam.Open()
cam.StartGrabbing()
with cam.RetrieveResult(2000) as result:

    # Calling AttachGrabResultBuffer creates another reference to the
    # grab result buffer. This prevents the buffer's reuse for grabbing.
    img.AttachGrabResultBuffer(result)

    filename = "saved_pypylon_img-{:}.png".format(time.strftime("%H:%M:%S"))
    img.Save(pylon.ImageFileFormat_Png, filename)

    # In order to make it possible to reuse the grab result for grabbing
    # again, we have to release the image (effectively emptying the
    # image object).
    img.Release()

cam.StopGrabbing()
cam.Close()