import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Colors
blue = (255,0,0)
green = (0, 255, 0)
white = (255,255,255)
red = (0,0,255)

# Models , classes and coordinates
pickleFile = r'coordinate'
class_file = r'models\main\class.txt'
model_file = r'models\main\best.pt'
vid_path = r'data\cctv_trimmed.mp4'

model=YOLO(model_file)
classes = open(class_file, "r")
classes = classes.read()
classes = classes.split("\n")

def calculate_distance(object_height, focal_length, apparent_height):
    distance = (object_height * focal_length) / apparent_height
    return distance

def calculate_focus(object_height , distance , apparent_height):
    focal_length = (distance * apparent_height) 
    focal_length = focal_length / object_height
    return focal_length

def main():
    frame_skip_count = 0
    # Initialize the camera
    cap = cv2.VideoCapture(vid_path)
    
    # Assuming known parameters (focal length, object width)
    object_height = 142     # Actual width of the object in meters
    distance = 1560
    focal_length = 0

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        
        frame_skip_count += 1
        if frame_skip_count < 10:  
            continue
        else:
            frame_skip_count = 0

        # frame=cv2.resize(frame,(1020,772))
        # h , w , _ = frame.shape
        # frame = frame[187 : h ,254:w ]

        results=model.predict(frame)
        a = results
        a=a[0].boxes.data
        px=pd.DataFrame(a).astype("float")
        for index ,row in px.iterrows():
        
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=classes[d]
            if 'car' in c:
                W=int(x2 - x1)
                H=int(y2 - y1)
                # Get the apparent width of the object (you need to manually measure this from the frame)
                apparent_height = H  # Apparent width of the object in pixels
                cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)
                # Calculate the distance to the object
                # distance_calculated = calculate_distance(object_height, focal_length, apparent_height)
                # distance = distance_calculated
                focal_length = calculate_focus(object_height , distance , apparent_height)
        # Print the distance
        cv2.putText(frame , f"dist : {focal_length} cm" ,(100 , 150) ,cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0) , 2) 
        cv2.imshow("Output" , frame)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
