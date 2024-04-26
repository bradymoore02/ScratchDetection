import cv2
import cv2
import time
import numpy as np

# Function to process video frames
def process_frame(frame):
    # Crop the frame to the specified region of interest (ROI)
    x1,x2,y1,y2 = 300,1200,900,1350
    frameroi = frame[x1:x2,y1:y2]
    #frameroi = frame
    

    # Convert the cropped frame to grayscale
    gray1 = cv2.cvtColor(frameroi, cv2.COLOR_BGR2GRAY)
    #gray1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    
    
    edges1 = cv2.Canny(gray1, 60, 120, 15)

    # Define a kernel for dilation
    kernel = np.ones((6, 6), np.uint8)  # You can adjust the size of the kernel

    # Perform dilation on the edges
    dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)
    
    contours1, _ = cv2.findContours(dilated_edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_with_contours = gray1.copy()
    color = (255, 0, 0)  # Blue color in BGR format
    thickness = 2  # Thickness of the rectangle border
    cv2.rectangle(frame, (x1, y1), (x2,y2), color, thickness)
    cv2.drawContours(frame_with_contours, contours1, -1, (0, 255, 0), 1)
    
    for contour in contours1:
            if cv2.arcLength(contour, True) > 300:  # Use arcLength to get the length of the contour
                #cv2.drawContours(frame_with_contours[500:800,800:], [contour], -1, (0, 0, 0), 5)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
    return frame_with_contours


def new_Process(image):
    x1,x2,y1,y2 = 10,3000,1000,2000
    image = frame[x1:x2,y1:y2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    gSize = 15
    gray = cv2.GaussianBlur(gray, (gSize, gSize), gSize/2)

    

    # Find gradient magnitude using Sobel operator
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mg = np.sqrt(gradient_x**2 + gradient_y**2)

    # Threshold gradient magnitude with a threshold of 30 percentile of max value
    threshold = np.percentile(mg, 99)
    mgBw = (mg > threshold).astype(np.uint8) * 255

    # Apply morphological operation closing of binary image by a disk mask of 3 X 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)
    # Again close image for joining lines together
    mgBw = cv2.morphologyEx(mgBw, cv2.MORPH_CLOSE, kernel)  

    
    
        # Apply Canny edge detection
    edges = cv2.Canny(mgBw, 5, 30)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_edges = image.copy()

    # Dilate the edges to enhance them
    kernel = np.ones((7,7), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # Draw bounding rectangles around large contours
    
    for contour in contours:
        if cv2.arcLength(contour, True) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the processed image
    cv2.imshow('Processed Image', image)
    
    return image


# Open a video capture object (use 0 for the default camera)
cap = cv2.VideoCapture("StarfireTestVideos/H.mov")
width = int(cap.get(3))
height = int(cap.get(4))

print(width, height)
fps = cap.get(5)
# Check if the video capture object is successfully opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a video writer object
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'h264')  # Define the codec (codec depends on the file extension)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    # Read a frame from the camera
    start_time = time.time()
    ret, frame = cap.read()

    # If the frame is not read successfully, break the loop
    if not ret:
        break
    time.sleep(0.2)
    # Process the frame to find and draw contours in the ROI
    #blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    result_frame = new_Process(frame)

    # Display the result
    cv2.imshow('Contours in ROI', result_frame)
    out.write(result_frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(elapsed_time)

# Release the video capture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
