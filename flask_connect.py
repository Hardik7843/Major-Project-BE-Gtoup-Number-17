import cv2
from flask import Flask, render_template, Response
import pandas as pd
import numpy as np
from ultralytics import YOLO
import pickle



pickleFile = r'coordinate_cam'
class_file = r'coco.txt'
model_file = r'models\best_model.pt'
vid_path = r'data\parking.mp4'

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


app = Flask(__name__ , template_folder='templates')
cap = cv2.VideoCapture(vid_path) # Replace with your CCTV camera IP address

def process_frame(frame):
    # Add your processing logic here
    # For example, you can use OpenCV functions to manipulate the frame
    # You may want to use a separate function for more complex processing
    img=cv2.resize(img,(1020,772))
    h , w , _ = img.shape

    frame = img[0 : h ,110:w ]
    results=model.predict(frame)
    a = results
    a=a[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    count = 0
    for _,row in px.iterrows():
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
            cv2.circle(frame,(cx,cy),3,blue,-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)

            for area in poslist:
                i = i+1
                # print(f"ith test : {i}")
                PolyGoneTest=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),True)
                # cv2.putText(frame,f"D:{PolyGoneTest}",(cx,cy),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
                if  PolyGoneTest < 0 and PolyGoneTest > -20:
                    print(f"Red poly: {PolyGoneTest}")
                    cv2.rectangle(frame,(x1,y1),(x2,y2),white,2)
                    cv2.circle(frame,(cx,cy),3,blue,-1)
                    count = count + 1
                    cv2.polylines(frame,[np.array(area,np.int32)],True,red,2)
                else:
                    print(f"Green poly: {PolyGoneTest}")
                    cv2.polylines(frame,[np.array(area,np.int32)],True,green,2)
    
    space=(len(poslist)-count)

    count = 0
    cv2.putText(frame,str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    
    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_frame(frame)

        
        _, jpeg = cv2.imencode('.jpg', processed_frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('templates\index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
