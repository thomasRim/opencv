You need to edit the .cfg to fit your needs based on your object detector. Open it up in a code or text editor to do so.

-

I recommend having **batch** = 64 and **subdivisions** = 16 for ultimate results. If you run into any issues then up **subdivisions** to 32.

Make the rest of the changes to the cfg based on how many **classes** you are training your detector on.

Note: I set my **max_batches** = 6000, **steps** = 4800, 5400, I changed the **classes** = 1 in the *three* YOLO layers and **filters** = 18 in the *three convolutional* layers *before* the YOLO layers.

-

###How to Configure Your Variables:

**width** = 416

**height** = 416 (these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)

**max_batches** = (# of classes) * 2000 (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)

**steps** = (80% of max_batches), (90% of max_batches) (so if your **max_batches** = 10000, then **steps** = 8000, 9000)

**filters** = (# of classes + 5) * 3 (so if you are training for one class then your **filters** = 18, but if you are training for 4 classes then your **filters** = 27)

**Optional**: If you run into memory issues or find the training taking a super long time. In each of the three yolo layers in the cfg, change one line from random = 1 to random = 0 to speed up training but slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

-

#After Learning

To use .cfg file for detecting one should update  
**batch** = 64 to **batch** = 1  
 and  
 **subdivisions** = 16 (32) to **subdivisions** = 1