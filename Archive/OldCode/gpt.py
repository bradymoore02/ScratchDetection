import cv2
import cv2
import time
import numpy as np

# Function to process video frames
def process_frame(frame, roi_x, roi_y, roi_width, roi_height):
    # Crop the frame to the specified region of interest (ROI)
    roi1 = frame[:, 300:620]
    roi2 = frame[:, 620:940]
    roi3 = frame[:, 940:1260]

    # Convert the cropped frame to grayscale
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
    
    
    edges1 = cv2.Canny(gray1, 50, 150,9)
    edges2 = cv2.Canny(gray2, 100, 200)
    edges3 = cv2.Canny(gray3, 50, 150)

    # Define a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)  # You can adjust the size of the kernel

    # Perform dilation on the edges
    dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)
    
    contours1, _ = cv2.findContours(dilated_edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours3, _ = cv2.findContours(edges3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    #print(contours1)
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours[:, 300:620], contours1, -1, (0, 255, 0), 1)
    cv2.drawContours(frame_with_contours[:, 620:940], contours2, -1, (0, 0, 255), 1)
    cv2.drawContours(frame_with_contours[:, 940:1260], contours3, -1, (255, 0, 0), 1)

    for contour in contours1:
            if cv2.contourArea(contour) > 100:  # Use arcLength to get the length of the contour
                print()
                cv2.drawContours(frame_with_contours[:, 300:620], [contour], -1, (0, 0, 0), 5)

    # Specify the vertical line parameters
    line_color = (0, 0, 0)  # Black color
    line_thickness = 2
    line_positions = [0,320, 640, 960]  # List of vertical line positions

    # Draw black vertical lines on the image
    for x in line_positions:
        x = x+300
        cv2.line(frame_with_contours, (x, 0), (x, height), line_color, line_thickness)

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

# Specify the region of interest (ROI) parameters
roi_x, roi_y, roi_width, roi_height = 0, 400, 2000, 200

while cap.isOpened():
    # Read a frame from the camera
    start_time = time.time()
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break

    # Process the frame to find and draw contours in the ROI
    #blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    result_frame = process_frame(frame, roi_x, roi_y, roi_width, roi_height)

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
