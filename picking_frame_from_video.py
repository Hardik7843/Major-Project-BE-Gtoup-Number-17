import cv2

def save_frame(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return
    

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break
        
        # frame=cv2.resize(frame,(1020,772))
        # h , w , _ = frame.shape
        # frame = frame[187 : h ,254:w ]

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save the frame
            cv2.imwrite(output_path, frame)
            print(f"Frame saved as {output_path}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'data\main.mp4'  # Change this to your video file path
    output_path = r'empty.jpg'  # Change this to your desired output path
    
    save_frame(video_path, output_path)
