import cv2
import numpy as np
import picamera2
import RPi.GPIO as gp
import os
import time

### Initialize GPIO pins and multicamera adapter ###
gp.setwarnings(False)
gp.setmode(gp.BOARD)

gp.setup(7, gp.OUT)
gp.setup(11, gp.OUT)
gp.setup(12, gp.OUT)

i2c = "i2cdetect -y 1"
os.system(i2c)

picam2=picamera2.Picamera2()

def takePhoto(camera):
    if camera == 'A':
        i2c = "i2cset -y 1 0x70 0x00 0x04"
        os.system(i2c)
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)
    elif camera == 'B':
        i2c = "i2cset -y 1 0x70 0x00 0x05"
        os.system(i2c)
        gp.output(7, True)
        gp.output(11, False)
        gp.output(12, True)
    elif camera == 'C':
        i2c = "i2cset -y 1 0x70 0x00 0x06"
        os.system(i2c)
        gp.output(7, False)
        gp.output(11, True)
        gp.output(12, False)

    picam2.start()
    im = picam2.capture_array()
    picam2.stop()
    return im


def previewVideo(camera):
    if camera == 'A':
        i2c = "i2cset -y 1 0x70 0x00 0x04"
        os.system(i2c)
        gp.output(7, False)
        gp.output(11, False)
        gp.output(12, True)
    elif camera == 'B':
        i2c = "i2cset -y 1 0x70 0x00 0x05"
        os.system(i2c)
        gp.output(7, True)
        gp.output(11, False)
        gp.output(12, True)
    elif camera == 'C':
        i2c = "i2cset -y 1 0x70 0x00 0x06"
        os.system(i2c)
        gp.output(7, False)
        gp.output(11, True)
        gp.output(12, False)

    picam2.start()
    while True:

        print('previewing')
        im = picam2.capture_array()
        im = im[0:3200,300:400]
        cv2.imwrite("h.jpg",im)
        # Display the captured image
        #cv2.imshow("Image", im)#
        time.sleep(1)
    picam2.stop()
    return 

previewVideo('B')
