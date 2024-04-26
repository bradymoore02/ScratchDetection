import cv2
import cv2
import time
import numpy as np

# Function to process video frames
def process_frame(frame):
    # Crop the frame to the specified region of interest (ROI)
    x1,x2,y1,y2 = 300,1200,900,1350
    frameroi = frame[x1:x2,y1:y2]
    
    im1 = cv2.resize(frameroi, None, fx=0.5, fy=0.5)
    cv2.imshow('Contours in ROI', im1)
    input("1")

    # Convert the images into grayscale
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Contours in ROI', gray)
    input("2")
    # Apply a Gaussian filter of size 15 X 15
    gSize = 15
    gray = cv2.GaussianBlur(gray, (gSize, gSize), gSize/2)
    cv2.imshow('Contours in ROI', gray)
    input("3")
    # Find gradient magnitude using Sobel operator
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mg = np.sqrt(gradient_x**2 + gradient_y**2)

    # Threshold gradient magnitude with a threshold of 30 percentile of max value
    threshold = np.percentile(mg, 30)
    mgBw = (mg > threshold).astype(np.uint8) * 255

    # Apply morphological operation closing of binary image by a disk mask of 3 X 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)

    # Apply particle analysis (connected component labeling)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mgBw, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 500:
            mgBw[labels == i] = 0

    # Again close image for joining lines together
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)

    # Fill holes in the image
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)

    # Final Annotations:
    # Overlay annotations on the original image for visualization
    final_annotations = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    return final_annotations


# Open a video capture object (use 0 for the default camera)
cap = cv2.VideoCapture("StarfireTestVideos/B.mov")
width = int(cap.get(3))
height = int(cap.get(4))

print(width, height)
fps = cap.get(5)
# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while cap.isOpened():
    # Read a frame from the camera
    start_time = time.time()
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break
    time.sleep(0.4)
    # Process the frame to find and draw contours in the ROI
    #blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    result_frame = process_frame(frame)

    # Display the result
    cv2.imshow('Contours in ROI', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)
    input("waiting")
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
