import cv2
import numpy as np

# Read an image file
image = cv2.imread("StarfireTestVideos/shot.png")

# Check if the image was successfully read
if image is not None:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)
    cv2.waitKey(0)  # Wait indefinitely for a key press

    gSize = 15
    gray = cv2.GaussianBlur(gray, (gSize, gSize), gSize/2)
    cv2.imshow('Grayscale Image', gray)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    

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
    cv2.imshow('Grayscale Image', mgBw)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    
    
        # Apply Canny edge detection
    edges = cv2.Canny(mgBw, 5, 30)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_edges = image.copy()
    cv2.drawContours(image_with_edges, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Grayscale Image', image_with_edges)
    cv2.waitKey(0)  # Wait indefinitely for a key press

    # Dilate the edges to enhance them
    kernel = np.ones((9,9), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Grayscale Image', image_with_contours)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    # Draw bounding rectangles around large contours
    
    for contour in contours:
        if cv2.arcLength(contour, True) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the processed image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()
else:
    print("Error: Could not read the image file or the file does not exist.")
