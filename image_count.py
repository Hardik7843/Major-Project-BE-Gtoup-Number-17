import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle

pickleFile = r'coordinate'
class_file = r'coco.txt'
model_file = r'yolov8s.pt'
img_path = r'data\cars.jpg'

model=YOLO(model_file)
classes = open(class_file, "r")
classes = classes.read()
classes = classes.split("\n")

try:
    with open(pickleFile , 'rb') as f:
        poslist = pickle.load(f)

except:
    poslist = []

# Colors
blue = (255,0,0)
green = (0, 255, 0)
white = (255,255,255)
red = (0,0,255)

while True:    
    frame = cv2.imread(img_path)
    frame=cv2.resize(frame,(1020,772))

    results=model.predict(frame)
    a = results
    a=a[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    count = 0
    list = []
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=classes[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2

        
            for area in poslist:
                PolyGoneTest=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
                
                if PolyGoneTest>=0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)
                    cv2.circle(frame,(cx,cy),3,blue,-1)
                    list.append(c)
                    A = len(list)
                    count = A
                    cv2.polylines(frame,[np.array(area,np.int32)],True,red,2)
                else:
                    cv2.polylines(frame,[np.array(area,np.int32)],True,green,2)
                
    print(count)

    space=(len(poslist)-count)

    count = 0
    cv2.putText(frame,str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(0)&0xFF==27:
        break
# cap.release()
cv2.destroyAllWindows()


# Connecting MySQL server

# # host = "127.0.0.1"
# port = "3306"
# user_name = "root"
# pwd = "root"
# dbname = "park"
# mydb = mysql.connector.connect(
# #   host= host,
#   user = user_name,
#   password=pwd,
# #   database= dbname
# )
# print(mydb.is_connected())
    


# # Putting cursor in MySQL
# cursorObject = mydb.cursor()



