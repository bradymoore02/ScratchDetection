import cv2

image = cv2.imread('clad.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny edge detection
edges = cv2.Canny(gray, 50, 100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours with different colors based on strength
for contour in contours:
    # Calculate the area of the contour (you can use other criteria for classification)
    area = cv2.contourArea(contour)

    # Assign colors based on the area (you can customize this based on your criteria)
    if area < 1000:
        color = (0, 0, 255)  # Red for weak contours
    else:
        color = (255, 0, 0)  # Blue for strong contours

    # Draw the contour with the specified color
    cv2.drawContours(image, [contour], 0, color, 2)

# Display the results
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()