import cv2
import numpy as np
import pickle

imgPath = r'data\cars.jpg'
pickleFile = r'coordinate'

try:
    with open(pickleFile, 'rb') as f:
        posList = pickle.load(f)
        print(posList[1][0])    
except:
    posList = []

area = []
def RGB(event, x, y, flags, area):
    if area == None:
        area = []

    if event == cv2.EVENT_LBUTTONDOWN :  
        colorsBGR = (x, y)
        area.append(colorsBGR)
        print(f"(x,y): {colorsBGR}")
        return area
    
    if event == cv2.EVENT_RBUTTONDOWN:
        posList.append(area)
        print(f"area appended: {area}")
        
       
    

cv2.namedWindow('Image')

while True:    
    frame = cv2.imread(imgPath)
    frame=cv2.resize(frame,(1020,772))

    for ara in posList:
        print(f"area from poslist: {ara}")
        cv2.polylines(frame,[np.array(ara,np.int32)],True,(0,255,0),2)

    cv2.imshow("Image", frame)
    cv2.setMouseCallback("Image", RGB , area)
    print(f"area returned : {area}")

    if len(area) == 4:
        ar = []
        area = ar

    if cv2.waitKey(0)&0xFF==27:
        break

print(f"poslist: {posList}")
with open(pickleFile, 'wb') as f:
        pickle.dump(posList, f) 
cv2.destroyAllWindows()



    




