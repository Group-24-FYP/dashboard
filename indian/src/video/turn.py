import cv2

# Open the video file
video_path = 'toothbrush_conveyorbelt_R.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output video file name and codec
output_path = 'rotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Read and process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Rotate the frame by 90 degrees counterclockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Write the rotated frame to the output video file
        output.write(rotated_frame)

        # Display the rotated frame (optional)
        cv2.imshow('Rotated Frame', rotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and output writer
cap.release()
output.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
