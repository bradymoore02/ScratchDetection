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

### Camera settings configuration ###
picam2 = picamera2.Picamera2()
config = picam2.create_still_configuration({"size": (3280, 2464)})
picam2.set_controls({'ExposureTime': 300, "AnalogueGain": 3.0})
picam2.align_configuration(config)
picam2.configure(config)
#picam2.set_controls({'ExposureTime': 50000, "AnalogueGain": 1.0})

def analyze(image):
    y1,y2,x1,x2 = 1630,1810,0, 2000
    image = image[x1:x2,y1:y2]  
    if clean:
        return image
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gSize = 15
    gray = cv2.GaussianBlur(gray, (gSize, gSize), gSize/2)
    

    # Find gradient magnitude using Sobel operator
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mg = np.sqrt(gradient_x**2 + gradient_y**2)

    # Threshold gradient magnitude with a threshold of 30 percentile of max value
    threshold = np.percentile(mg, 97)
    mgBw = (mg > threshold).astype(np.uint8) * 255

    # Apply morphological operation closing of binary image by a disk mask of 3 X 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)
    # Again close image for joining lines together
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)  
    
    
        # Apply Canny edge detection
    edges = cv2.Canny(mgBw, 5, 30)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_edges = image.copy()
    cv2.drawContours(image_with_edges, contours, -1, (0, 255, 0), 1)

    # Dilate the edges to enhance them
    kernel = np.ones((9,9), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
    #cv2.imshow('Grayscale Image', image_with_contours)
    # Draw bounding rectangles around large contours
    
    for contour in contours:
        if cv2.arcLength(contour, True) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

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

for i in range(0,100):
    clean = True
    im = takePhoto('A')
    im_analyzed = analyze(im)
    cv2.imwrite(f'out5/a{i}.jpg', im_analyzed)

    im = takePhoto('B')
    im_analyzed = analyze(im)
    cv2.imwrite(f'out5/b{i}.jpg', im_analyzed)

    im = takePhoto('C')
    im_analyzed = analyze(im)
    cv2.imwrite(f'out5/c{i}.jpg', im_analyzed)
