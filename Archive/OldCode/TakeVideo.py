import cv2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
#fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec

out = cv2.VideoWriter('StarfireTestVideos/test.mp4', fourcc, 20.0, (640, 480))  # ('output_file.mp4', codec, fps, (width, height))

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Write the frame to the output video file
    out.write(frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
video_capture.release()
out.release()
cv2.destroyAllWindows()