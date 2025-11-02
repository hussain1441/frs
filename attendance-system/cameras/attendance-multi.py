# multi-camera attendance system
import cv2
import insightface
from pymongo import MongoClient
import numpy as np
import threading
import time
from datetime import date, datetime
from attendance import multi_camera_attendance, ensure_entry, get_last_punch_time
from dotenv import load_dotenv
import os

load_dotenv()

latest_frames = {}  # camera_name -> latest frame
frame_locks = {}  # camera_name -> threading.Lock()

# MongoDB connection
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

# Load model
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # USE ctx_id=0 FOR GPU

# Camera configs
cameras = [
    {
        "name": "PunchIn",
        "rtsp_url": "rtsp://admin:admin@192.168.1.2:8554/live",
        "line_start": (448, 18),
        "line_end": (517, 1066),
    },
    {
        "name": "PunchOut",
        "rtsp_url": "rtsp://localhost:8554/mystream",
        "line_start": (10, 290),
        "line_end": (590, 275),
    },
]

# Cache embeddings
db_embeddings = []
for user in collection.find():
    db_embeddings.append(
        {
            "name": user["name"],
            "roll_number": user["roll_number"],
            "embedding": np.array(user["embedding"]),
        }
    )
print(f"Successfully cached {len(db_embeddings)} embeddings in MongoDB")


# Utils
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Frame reader for a camera
def frame_reader(cam_config, frame_dict, lock):
    capture = cv2.VideoCapture(cam_config["rtsp_url"])
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, 30)

    while True:
        gotFrame, frame = capture.read()
        if not gotFrame:
            print(f"❌ Did not get frame from {cam_config['name']}")
            time.sleep(0.1)
            continue

        with lock:
            frame_dict["frame"] = frame.copy()


# Face recognition + line check for a frame
def process_frame(frame, cam_config):
    faces = model.get(frame)

    # Draw the camera line always
    cv2.line(frame, cam_config["line_start"], cam_config["line_end"], (0, 0, 255), 2)

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

        # bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, face_center, 5, (255, 0, 0), -1)

        if best_score > 0.6:
            person_id = best_match["roll_number"]
            person_name = best_match["name"]
            color = (0, 255, 0)
            label = f"{best_match['name']}"
            print(
                f"🟩 Recognized: {person_name} ({person_id}) | Score: {best_score:.2f}"
            )

            # Check if face center crosses the line
            x1_line, y1_line = cam_config["line_start"]
            x2_line, y2_line = cam_config["line_end"]
            if x2_line - x1_line != 0:
                m = (y2_line - y1_line) / (x2_line - x1_line)
                b = y1_line - m * x1_line
                y_on_line = m * face_center[0] + b
                if face_center[1] > y_on_line:
                    now = time.time()
                    last_punch_time = get_last_punch_time(person_id)
                    if last_punch_time is None or (now - last_punch_time) > 10:
                        ensure_entry()
                        multi_camera_attendance(
                            person_id, person_name, cam_config["name"]
                        )
        else:
            color = (0, 0, 255)
            label = f"Unknown - {best_score:.2f}"
            print(f"🟥 Unknown face | Best score: {best_score:.2f}")

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
        cv2.putText(
            frame,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return frame


# Main loop for a camera
def camera_loop(cam_config):
    lock = threading.Lock()
    frame_locks[cam_config["name"]] = lock
    latest_frames[cam_config["name"]] = None

    def reader():
        capture = cv2.VideoCapture(cam_config["rtsp_url"])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, 30)

        while True:
            gotFrame, frame = capture.read()
            if not gotFrame:
                print(f"❌ Did not get frame from {cam_config['name']}")
                time.sleep(0.1)
                continue
            with lock:
                latest_frames[cam_config["name"]] = frame.copy()

    threading.Thread(target=reader, daemon=True).start()


# Run all cameras
def main():
    # Start all camera threads
    for cam in cameras:
        camera_loop(cam)

    print("Starting multi-camera face recognition...")
    print("Press 'q' to quit all windows.")

    processing_interval = 0.3
    last_process_time = time.time()

    while True:
        current_time = time.time()
        if current_time - last_process_time >= processing_interval:
            for cam in cameras:
                name = cam["name"]
                with frame_locks[name]:
                    frame = latest_frames[name]
                    if frame is not None:
                        processed = process_frame(frame, cam)
                        cv2.imshow(f"Camera: {name}", processed)
            last_process_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
