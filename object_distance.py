import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle
import json




"""Colors"""
blue = (255,0,0)
green = (0, 255, 0)
white = (255,255,255)
red = (0,0,255)

"""File Reading Sections"""
pickleFile = r'coordinate'
class_file = r'models\main\class.txt'
model_file = r'models\main\best.pt'
vid_path = r'data\cctv_trimmed.mp4'
dist_file = r'distances.json'

model=YOLO(model_file)
classes = open(class_file, "r")
classes = classes.read()
classes = classes.split("\n")


try:
    with open(dist_file, 'w+') as file:
        file.seek(0)  # Move the cursor to the beginning of the file
        distances = json.load(file)
except:
    distances = {}

try:
    with open(pickleFile , 'rb') as f:
        poslist = pickle.load(f)

except:
    poslist = []


def calculate_distance(object_height, focal_length, apparent_height):
    distance = (object_height * focal_length) 
    distance = distance / apparent_height
    return distance

    # Initialize the camera
cap = cv2.VideoCapture(vid_path)

# Assuming known parameters (focal length, object width)
focal_length = 1472  # Focal length in pixels (this value needs to be determined/calibrated)
object_height = 142     # Actual width of the object in meters
distance = "Null"
frame_skip_count = 0
count = 0
while True:
    ret,frame = cap.read()
    if not ret:
        break

    frame_skip_count += 1
    if frame_skip_count < 10:  
        continue
    else:
        frame_skip_count = 0

    frame=cv2.resize(frame,(1020,772))
    h , w , _ = frame.shape
    frame = frame[187 : h ,254:w ]

    results=model.predict(frame)
    a = results
    a=a[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    count = 0
    distances = {}
    for area_number , area in enumerate(poslist):
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
                area_P = ((x1 , y1 ),(x2 , y1) ,(x2 , y2) , (x1 , y2) )
                xy_i = area[0]
                xy_f = area[2]
                cx = int((xy_f[0] + xy_i[0])/2)
                cy = int((xy_f[1] + xy_i[1])/2)

                PolyGoneTest=cv2.pointPolygonTest(np.array(area_P,np.int32),((cx,cy)),True)
                apparent_height = H  
                distance_calculated = calculate_distance(object_height, focal_length, apparent_height)
                distance = distance_calculated
                distances[(area_number)] = distance
                # print(f"area_P: {area_P}")
                # print(f"area : {area}")
                # print(f"bbox:((x1,y1),(x2,y2)): {((x1,y1),(x2,y2))}")
                # print(f"cx,cy: {cx , cy}")
                # print(f"xy_f[0] - xy_i[0] = {xy_f[0] - xy_i[0]}")
                # cv2.circle(frame,(cx,cy),3,blue,-1)
                if distance >= 2000 and distance <= 2300:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)
                else:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),red,2)
                if PolyGoneTest>=25:
                    print(f"polydist: {PolyGoneTest}")
                    cv2.circle(frame,(cx,cy),3,blue,-1)
                    cv2.polylines(frame,[np.array(area,np.int32)],True,red,2)
                    count = count + 1
                else:
                    cv2.polylines(frame,[np.array(area,np.int32)],True,green,2)
            
                
            
    # Print the distance
    cv2.putText(frame , f"dist : {distance} cm" ,(100 , 150) ,cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0) , 2) 
    cv2.imshow("Output" , frame)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break


with open(dist_file, 'w') as file:
    json.dump(distances, file)
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
