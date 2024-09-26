import cv2
import pickle

# Path to the binary file
pickleFile = r'slot_number'
vid_path = r'data\Fully_parked.mp4'

points = []

def mouseClick(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        print(f"Point ({x}, {y}) added.")
        with open(pickleFile, 'wb') as f:
            pickle.dump(points , f)

try:
    with open(pickleFile, 'rb') as f:
        points = pickle.load(f)
        print("Existing points loaded:", points)
except:
    points = []
        
cap = cv2.VideoCapture(vid_path)

while True:
    ret,frame = cap.read()
    frame=cv2.resize(frame,(1020,772))
    h , w , _ = frame.shape
    frame = frame[250 : h ,490:w-170]
    
    cv2.imshow("Image", frame)
    cv2.setMouseCallback("Image", mouseClick)
    
    for point in points:
        x , y = point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Image" , frame)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        cv2.imwrite("Image1.jpg", frame)  # Press 'Esc' key to exit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
