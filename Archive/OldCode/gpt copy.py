import cv2
import cv2
import time
import numpy as np

# Function to process video frames
def process_frame(frame):
    # Crop the frame to the specified region of interest (ROI)

    frameroi = frame[500:800,800:]
    frameroi = frame

    # Convert the cropped frame to grayscale
    gray1 = cv2.cvtColor(frameroi, cv2.COLOR_BGR2GRAY)

    
    
    edges1 = cv2.Canny(gray1, 40, 150, 5)

    # Define a kernel for dilation
    kernel = np.ones((6, 6), np.uint8)  # You can adjust the size of the kernel

    # Perform dilation on the edges
    dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)
    
    contours1, _ = cv2.findContours(dilated_edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    #print(contours1)
    frame_with_contours = frameroi.copy()
    cv2.drawContours(frame_with_contours, contours1, -1, (0, 255, 0), 1)

    for contour in contours1:
            if cv2.arcLength(contour, True) > 300:  # Use arcLength to get the length of the contour
                #cv2.drawContours(frame_with_contours[500:800,800:], [contour], -1, (0, 0, 0), 5)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)


    return frame_with_contours


# Open a video capture object (use 0 for the default camera)
cap = cv2.VideoCapture(0)
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

    # Process the frame to find and draw contours in the ROI
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    result_frame = process_frame(blurred_frame)

    # Display the result
    cv2.imshow('Contours in ROI', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
