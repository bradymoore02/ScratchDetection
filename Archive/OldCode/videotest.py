import cv2

cap = cv2.VideoCapture('IMG_4436.MOV')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, 7)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    cv2.drawContours(frame, contours, 0, (255, 0, 0), 2)
    
    cv2.imshow('Edge Detection', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
