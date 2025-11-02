# real time recognition with separate threads (FAST)

import cv2
import threading
import time
import numpy as np
from pymongo import MongoClient
import insightface
from dotenv import load_dotenv
import os

load_dotenv()

# --- MongoDB setup (same as before) ---
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

# cache embeddings
db_embeddings = []
for user in collection.find():
    db_embeddings.append(
        {
            "name": user["name"],
            "roll_number": user["roll_number"],
            "embedding": np.array(user["embedding"]),
        }
    )
print(f"Cached {len(db_embeddings)} embeddings ✅")

# --- Model setup ---
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # USE ctx_id=0 FOR GPU
print("Model loaded ✅")

# --- Shared state ---
latest_frame = None
latest_results = []
lock = threading.Lock()
stop_flag = False


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Camera capture thread ---
def camera_loop():
    global latest_frame, stop_flag
    cap = cv2.VideoCapture(1)
    # rtsp_url = "rtsp://localhost:8554/mystream"
    # cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            latest_frame = frame.copy()
    cap.release()


# --- Recognition thread ---
def recognition_loop():
    global latest_frame, latest_results, stop_flag
    while not stop_flag:
        time.sleep(0.3)  # process every ~300 ms
        with lock:
            if latest_frame is None:
                continue
            frame_copy = latest_frame.copy()
        faces = model.get(frame_copy)
        new_results = []
        for face in faces:
            embedding = face.normed_embedding.astype(float).tolist()
            best_match = None
            best_score = -1
            for user in db_embeddings:
                score = cosine_similarity(embedding, user["embedding"])
                if score > best_score:
                    best_score = score
                    best_match = user
                if score > 0.85:
                    break
            bbox = face.bbox.astype(int)
            if best_score > 0.6:
                label = f"{best_match['name']} ({best_match['roll_number']})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)
            new_results.append((bbox, label, color))
        with lock:
            latest_results = new_results


def draw_detections(frame, results, corner=False, corner_length=20):
    """Overlay detection results (bounding boxes + labels) onto the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        results (list): List of (bbox, label, color) tuples.
        corner (bool): If True, draw only corner lines instead of full rectangles.
        corner_length (int): Length of each corner line.
    """
    for bbox, label, color in results:
        x1, y1, x2, y2 = bbox

        if corner:
            # Draw corner-only bounding box
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
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


# --- Start threads ---
camera_thread = threading.Thread(target=camera_loop, daemon=True)
recognition_thread = threading.Thread(target=recognition_loop, daemon=True)
camera_thread.start()
recognition_thread.start()

# --- Display loop (main thread) ---
while True:
    with lock:
        frame = None if latest_frame is None else latest_frame.copy()
        results = list(latest_results)
    if frame is not None:
        draw_detections(frame, results, corner=True)
        # overlay results
        # for bbox, label, color in results:
        #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        #     cv2.putText(
        #         frame,
        #         label,
        #         (bbox[0], bbox[1] - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (255, 255, 255),
        #         2,
        #     )
        cv2.imshow("Realtime Face Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        stop_flag = True
        break

cv2.destroyAllWindows()
