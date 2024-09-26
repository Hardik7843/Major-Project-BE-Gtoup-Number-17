import cv2
import pickle
import numpy as np

pickleFile = r'coordinate'
referen_image_path = r'reference_frame.jpg'
vid_path = r'data\Fully_parked.mp4'
# vid_path = r'data\bus.mov'
# vid_path = r'data\2 legal  1 parking + outgoing 1 illegal_part2.mp4'
# vid_path = r'data\main.mp4'

drawing = False

try:
    with open(pickleFile, 'rb') as f:
        posList = pickle.load(f)
        print(posList[1][0])    
except:
    posList = []


def mouseClick(events, x, y, flags, params):
    global start_x, start_y, end_x, end_y, drawing
    if events == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        print("Start", x , y)
        drawing = True   
    elif events == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x,y
        drawing = False
        print("End", x, y)
        pt_1_x , pt_1_y = end_x , start_y
        pt_2_x , pt_2_y = start_x  ,end_y
        posList.append(((start_x, start_y),(pt_1_x , pt_1_y) , (end_x, end_y) , (pt_2_x , pt_2_y)))
    
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            start_pt , _ , end_pt , _ = pos
            x1 ,y1 = start_pt
            x2, y2 = end_pt
            if x1 < x < x2 and y1 < y < y2:
                posList.pop(i)

    with open(pickleFile, 'wb') as f:
        pickle.dump(posList, f)

cap = cv2.VideoCapture(vid_path)
frame_skip_count = 0
while True:
    frame_skip_count += 1
    if frame_skip_count < 20:  
        continue
    else:
        frame_skip_count = 0

    ret,frame = cap.read()
    frame=cv2.resize(frame,(1020,772))
    h , w , _ = frame.shape
    frame = frame[250 : h ,490:w-170]

    for ara in posList:
        print(f"area from poslist: {ara}")
        cv2.polylines(frame, [np.array(ara, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("Image", frame)

    cv2.setMouseCallback("Image", mouseClick)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        cv2.imwrite("Image1.jpg", frame)  # Press 'Esc' key to exit
        break

cv2.destroyAllWindows()
