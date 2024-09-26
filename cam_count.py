import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle
import time



"""File Reading Sections"""

# Models , classes and coordinates
pickleFile = r'coordinate'
class_file = r'models\main\class.txt'
model_file = r'models\main\best.pt'
vid_path = r'data\cctv_trimmed.mp4'

model=YOLO(model_file)
classes = open(class_file, "r")
classes = classes.read()
classes = classes.split("\n")

try:
    with open(pickleFile , 'rb') as f:
        poslist = pickle.load(f)

except:
    poslist = []


"""Colors"""
blue = (255,0,0)
green = (0, 255, 0)
white = (255,255,255)
red = (0,0,255)

"""Constants"""
K = 0   # Callibration Factor for disntance
focal_length = 1472  # Focal length in pixels (this value needs to be determined/calibrated)
object_height = 142     # Actual width of the object in meters
distance = "Null"
frame_skip_count = 0
count = 0

# Dictionary to keep track of slot statuses
slot_statuses = {slot_number: False for slot_number in range(1, len(poslist) + 1)}

def calculate_distance(object_height, focal_length, apparent_height):
    distance = (object_height * focal_length) 
    distance = distance / apparent_height
    return distance

cap=cv2.VideoCapture(vid_path)
frame_skip_count = 0
while True:  
    start = time.time()  
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
    space = 0
    count = 0
    list = []
    
    for area in poslist:
        for index,row in px.iterrows():
    
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=classes[d]
            if 'car' in c:
                W=int(x2 - x1)
                H=int(y2 - y1)

                apparent_height = H  
                distance_calculated = calculate_distance(object_height, focal_length, apparent_height)
                distance = distance_calculated - K 
                
                if distance < 1500:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
                    continue

                area_P = ((x1 , y1 ),(x2 , y1) ,(x2 , y2) , (x1 , y2) )
                xy_i = area[0]
                xy_f = area[2]
                cx = int((xy_f[0] + xy_i[0])/2)
                cy = int((xy_f[1] + xy_i[1])/2)
                PolyGoneTest=cv2.pointPolygonTest(np.array(area_P,np.int32),((cx,cy)),False)
                cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)
                if PolyGoneTest>=0:
                    count = count + 1
                    cv2.circle(frame,(cx,cy),3,blue,-1)
                    cv2.polylines(frame,[np.array(area,np.int32)],True,red,2)
                else:
                    cv2.polylines(frame,[np.array(area,np.int32)],True,green,2)        
            elif 'bus' in c or 'truck' in c or 'motorcycle' in c:
                W=int(x2 - x1)
                H=int(y2 - y1)

                apparent_height = H  
                distance_calculated = calculate_distance(object_height, focal_length, apparent_height)
                distance = distance_calculated - K 
                # print(f"distance : {distance}")
                if distance < 1500:
                    area_P = ((x1 , y1 ),(x2 , y1) ,(x2 , y2) , (x1 , y2) )
                    cv2.polylines(frame,[np.array(area_P,np.int32)],True,red,2)

    space=(len(poslist)-count)
    count = 0
    cv2.putText(frame,f"Avl: {space}",(23,30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow("RGB", frame)
    end = time.time()    
    print(f"Time to process: {end - start}")    
    
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


