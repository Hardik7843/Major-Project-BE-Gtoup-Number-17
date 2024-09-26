import cv2
import asyncio
import websockets
import base64

async def send_frames(websocket, path):
    cap = cv2.VideoCapture('data\main.mp4')
    while True:
        ret, frame = cap.read()
        # Process frame (if needed)
        # Convert frame to base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        # Send frame to WebSocket client
        await websocket.send(frame_base64)

# Start WebSocket server
start_server = websockets.serve(send_frames, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
