# previews the recognized faces on a local website (OPTIMIZED VERSION)

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

load_dotenv()

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
print("Model loaded ✅")

# cache embeddings
db_embeddings = []
for user in collection.find():
    emb = np.array(user["embedding"], dtype=np.float32)
    db_embeddings.append(
        {
            "name": user["name"],
            "roll_number": user["roll_number"],
            "embedding": emb / np.linalg.norm(emb),
        }
    )
print(f"Successfully cached {len(db_embeddings)} embeddings in MongoDB")

# global variables
latest_frame = None
latest_results = []
frame_lock = threading.Lock()
stop_flag = False


def cosine_similarity(a, b):
    return np.dot(a, b)  # Already normalized embeddings


def camera_loop():
    global latest_frame, stop_flag
    # Use camera (0 for default, 1 for external)
    cap = cv2.VideoCapture(0)
    # For RTSP stream, uncomment below:
    # rtsp_url = "rtsp://localhost:8554/mystream"
    # cap = cv2.VideoCapture(rtsp_url)

    # Optimize camera settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("❌ Did not get frame")
            time.sleep(0.1)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)  # ~30 fps

    cap.release()


def recognition_loop():
    global latest_frame, latest_results, stop_flag

    while not stop_flag:
        time.sleep(0.3)  # Process every ~300ms (like live-camera-mac.py)

        with frame_lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()

        # Face detection and recognition
        faces = model.get(frame_copy)
        new_results = []

        for face in faces:
            embedding = face.normed_embedding.astype(float).tolist()
            best_match = None
            best_score = -1

            # Find best match in database
            for user in db_embeddings:
                score = cosine_similarity(embedding, user["embedding"])
                if score > best_score:
                    best_score = score
                    best_match = user

                # Early break for high similarity
                if score > 0.85:
                    break

            bbox = face.bbox.astype(int)

            if best_score > 0.6:
                label = f"{best_match['roll_number']} - {best_score:.2f}"
                color = (0, 255, 0)  # green
                print(
                    f"🟩 Recognized: {best_match['name']} ({best_match['roll_number']}) | Score: {best_score:.2f}"
                )
            else:
                label = f"Unknown - {best_score:.2f}"
                color = (0, 0, 255)  # red
                print(f"🟥 Unknown face | Best score: {best_score:.2f}")

            new_results.append((bbox, label, color))

        with frame_lock:
            latest_results = new_results


def draw_detections(frame, results, corner=True, corner_length=20):
    """Overlay detection results with corner-style bounding boxes."""
    for bbox, label, color in results:
        x1, y1, x2, y2 = bbox

        if corner:
            # Draw corner-only bounding box (like live-camera-mac.py)
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 2)

            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 2)

            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 2)

            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 2)
        else:
            # Regular rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    return frame


async def send_frame(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.03)  # ~30 fps

            with frame_lock:
                if latest_frame is None or latest_results is None:
                    continue

                # Create a copy of the frame and draw detections
                frame_copy = latest_frame.copy()
                results_copy = list(latest_results)

            # Draw detections on the frame
            processed_frame = draw_detections(frame_copy, results_copy, corner=True)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode(
                ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
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
        <title>Face Recognition Stream (FULL SCREEN)</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body, html {
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: #000;
                font-family: Arial, sans-serif;
            }

            .container {
                width: 100vw;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background: #000;
                position: relative;
            }

            #videoCanvas {
                width: 100%;
                height: 100%;
                object-fit: contain;
                background: #000;
            }

            .status {
                position: absolute;
                top: 20px;
                left: 20px;
                color: #0f0;
                font-weight: bold;
                font-size: 18px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px 15px;
                border-radius: 8px;
                z-index: 100;
                backdrop-filter: blur(5px);
            }

            .controls {
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.7);
                padding: 10px 20px;
                border-radius: 8px;
                color: white;
                z-index: 100;
                backdrop-filter: blur(5px);
            }

            .fullscreen-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                border: 1px solid #444;
                padding: 10px 15px;
                border-radius: 8px;
                cursor: pointer;
                z-index: 100;
                backdrop-filter: blur(5px);
            }

            .fullscreen-btn:hover {
                background: rgba(50, 50, 50, 0.7);
            }

            @media (max-width: 768px) {
                .status {
                    font-size: 14px;
                    top: 10px;
                    left: 10px;
                }

                .controls {
                    bottom: 10px;
                    font-size: 14px;
                }

                .fullscreen-btn {
                    top: 10px;
                    right: 10px;
                    padding: 8px 12px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="status" id="status">🟢 CONNECTED</div>
            <button class="fullscreen-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <canvas id="videoCanvas"></canvas>
            <div class="controls" id="controls">
                Face Recognition Stream - <span id="fps">0 FPS</span>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');
            const fpsElement = document.getElementById('fps');
            let ws = new WebSocket("ws://" + location.host + "/ws");
            let lastFrameTime = Date.now();
            let frameCount = 0;
            let fps = 0;
            let fpsInterval = null;

            function updateFPS() {
                const now = Date.now();
                frameCount++;

                if (now - lastFrameTime >= 1000) {
                    fps = frameCount;
                    frameCount = 0;
                    lastFrameTime = now;
                    fpsElement.textContent = `${fps} FPS`;
                    status.textContent = `🟢 CONNECTED | ${fps} FPS`;
                }
            }

            function resizeCanvas() {
                // Keep canvas at full window size
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }

            function toggleFullscreen() {
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen().catch(err => {
                        console.log(`Error attempting to enable fullscreen: ${err.message}`);
                    });
                } else {
                    if (document.exitFullscreen) {
                        document.exitFullscreen();
                    }
                }
            }

            // Handle fullscreen changes
            document.addEventListener('fullscreenchange', () => {
                resizeCanvas();
            });

            // Handle window resize
            window.addEventListener('resize', resizeCanvas);

            ws.onmessage = (event) => {
                const img = new Image();
                img.src = "data:image/jpeg;base64," + event.data;
                img.onload = () => {
                    // Draw image centered and scaled to fit canvas
                    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                    const x = (canvas.width - img.width * scale) / 2;
                    const y = (canvas.height - img.height * scale) / 2;

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

                    updateFPS();
                };
            };

            ws.onopen = () => {
                console.log("✅ Connected to WebSocket");
                resizeCanvas();
                // Start FPS counter
                fpsInterval = setInterval(updateFPS, 1000);
            };

            ws.onclose = () => {
                console.log("❌ WebSocket closed");
                status.textContent = "🔴 DISCONNECTED - Reconnecting...";
                fpsElement.textContent = "0 FPS";
                if (fpsInterval) clearInterval(fpsInterval);

                setTimeout(() => {
                    ws = new WebSocket("ws://" + location.host + "/ws");
                }, 2000);
            };

            ws.onerror = () => {
                status.textContent = "🔴 CONNECTION ERROR";
            };

            // Initialize
            resizeCanvas();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await send_frame(websocket)


if __name__ == "__main__":
    # Start camera and recognition threads (like live-camera-mac.py)
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    recognition_thread = threading.Thread(target=recognition_loop, daemon=True)

    camera_thread.start()
    recognition_thread.start()

    print("🚀 Starting optimized face recognition server...")
    print("📡 Web interface available at: http://localhost:8000")
    print("🖥️  Displaying in FULL SCREEN mode")

    uvicorn.run(app, host="0.0.0.0", port=8000)
