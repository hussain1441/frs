# previews the recognized faces on a local website

import asyncio
import base64
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import insightface
import threading
import time
import numpy as np
from pymongo import MongoClient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
import os

# server
app = FastAPI()

# mongodb connection
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

# loading model
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # USE ctx_id=0 FOR GPU

# camera url
rtsp_url = "rtsp://localhost:8554/mystream"

# global variables
latest_frame = None
latest_processed_frame = None
frame_lock = threading.Lock()
processing = False

# cache embeddings
db_embeddings = []
for user in collection.find():
    emb = np.array(user["embedding"], dtype=np.float32)
    db_embeddings.append(
        {
            "name": user["name"],
            "roll_number": user["roll_number"],
            # "embedding": np.array(user["embedding"]),
            "embedding": emb / np.linalg.norm(emb),
        }
    )
print(f"Successfully cached {len(db_embeddings)} embeddings in MongoDB")


def cosine_similarity(a, b):
    score = np.dot(a, b)
    return score
    # a, b = np.array(a), np.array(b)
    # return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frame_reader():
    global latest_frame

    capture = cv2.VideoCapture(rtsp_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, 30)
    while True:
        for _ in range(1):
            gotFrame, frame = capture.read()
            if not gotFrame:
                break

        if not gotFrame:
            print("❌ Did not get frame")
            time.sleep(0.1)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)


def process_face(frame):
    # resize frame
    faces = model.get(frame)

    for face in faces:
        embedding = face.normed_embedding.astype(float).tolist()  # i embed here
        best_match = None
        best_score = -1

        for user in db_embeddings:
            score = cosine_similarity(embedding, user["embedding"])
            if score > best_score:
                best_score = score
                best_match = user

            # break early for high similarity
            if score > 0.85:
                break

        # bounding box
        bbox = face.bbox.astype(int)

        if best_score > 0.6:
            color = (0, 255, 0)  # green for recognized
            label = f"{best_match['roll_number']} - {best_score:.2f}"
            print(
                f"🟩 Recognized: {best_match['name']} ({best_match['roll_number']}) | Score: {best_score:.2f}"
            )
        else:
            color = (0, 0, 255)  # red for unknown
            label = f"Unknown - {best_score:.2f}"
            print(f"🟥 Unknown face | Best score: {best_score:.2f}")

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
        cv2.putText(
            frame,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    return frame


def face_processor():
    global latest_frame, latest_processed_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        processed = process_face(frame_copy)

        with frame_lock:
            latest_processed_frame = processed

        time.sleep(0.3)


async def send_frame(websocket: WebSocket):
    global latest_processed_frame, processing
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.03)  # ~30 fps
            with frame_lock:
                if latest_processed_frame is not None:
                    # and not processing:
                    # processing = True
                    # frame_to_process = latest_frame.copy()
                    # processing = False

                    # Process the frame
                    # processed_frame = process_face(frame_to_process)

                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(
                        ".jpg", latest_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    frame_bytes = buffer.tobytes()
                    # Send as base64 string
                    b64_bytes = base64.b64encode(frame_bytes).decode("utf-8")
                    await websocket.send_text(b64_bytes)
    except WebSocketDisconnect:
        print("Client Disconnected")


@app.get("/")
async def get_index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Stream</title>
        <style>
            body { background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; }
            canvas { border: 2px solid #444; border-radius: 8px; }
        </style>
    </head>
    <body>
        <canvas id="videoCanvas"></canvas>
        <script>
            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');
            const ws = new WebSocket("ws://" + location.host + "/ws");

            ws.onmessage = (event) => {
                const img = new Image();
                img.src = "data:image/jpeg;base64," + event.data;
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
            };

            ws.onopen = () => console.log("✅ Connected to WebSocket");
            ws.onclose = () => console.log("❌ WebSocket closed");
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await send_frame(websocket)


if __name__ == "__main__":
    threading.Thread(target=frame_reader, daemon=True).start()
    threading.Thread(target=face_processor, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
