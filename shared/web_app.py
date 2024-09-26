from flask import Flask, render_template
import asyncio
import websockets
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

async def receive_frames():
    async with websockets.connect('ws://localhost:8765') as websocket:
        while True:
            # Receive frame from WebSocket server
            frame_base64 = await websocket.recv()
            # Decode base64 string and convert to image
            buffer = base64.b64decode(frame_base64)
            nparr = np.frombuffer(buffer, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Display frame (if needed)
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(receive_frames())
