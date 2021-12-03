from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from connection_manager import ConnectionManager
from model import Model

app = FastAPI()
manager = ConnectionManager()

@app.get("/")
def basic():
    return "Hello from the backend! 👋"

@app.websocket("/run/{model_name}")
async def websocket_endpoint(websocket: WebSocket, model_name: str):

    # Determine the passed model, or raise exception
    model = Model(model_name)

    # Wait for client to connect
    await manager.connect(websocket)

    # For this demo project we can capture the webcam from the server-side.
    # Since the server is running on the local device, cv2 is able to access the webcam.
    # If this backend would be deployed to a cloud server, we will need to capture the webcam
    # on the client-side and send the encoded frames through this websocket.

    try:
        while True:
            payload = model.run()
            await manager.send_keypoints(payload, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
