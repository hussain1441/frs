# single camera attendence system

import cv2
import insightface
from pymongo import MongoClient
import numpy as np
import threading
import time
from datetime import date, datetime
from attendance import mark_attendance, ensure_entry, get_last_punch_time
from dotenv import load_dotenv
import os

load_dotenv()

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
frame_lock = threading.Lock()
processing = False
line_start = (5, 297)
line_end = (580, 284)

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
print(f"Successfully cached {len(db_embeddings)} embeddings in MongoDB")


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def frame_reader():
    global latest_frame

    capture = cv2.VideoCapture(rtsp_url)

    # add optional settings to downscale
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FPS, 30)  # try 15 if not working
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # loop to discard old frames
    # frame_count = 0
    while True:
        for _ in range(1):
            gotFrame, frame = capture.read()
            if not gotFrame:
                break

        if not gotFrame:
            print("❌ Did not get frame")
            time.sleep(0.1)
            continue

        # frame_count += 1
        # if frame_count % 5 == 0:
        #     with frame_lock:
        #         latest_frame = frame.copy()

        with frame_lock:
            latest_frame = frame.copy()

        # time.sleep(0.3)


def process_face(frame):
    global line_start, line_end

    faces = model.get(frame)

    for face in faces:
        embedding = face.normed_embedding.astype(float).tolist()  # i embed here

        # compare mongodb and current embedding

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
        x1, y1, x2, y2 = bbox
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # draw line (red) and face center (blue)
        cv2.circle(frame, face_center, 5, (255, 0, 0), -1)

        if best_score > 0.6:
            person_id = best_match["roll_number"]
            person_name = best_match["name"]

            color = (0, 255, 0)  # green for recognized
            label = f"{best_match['name']}"
            print(
                f"🟩 Recognized: {best_match['name']} ({best_match['roll_number']}) | Score: {best_score:.2f}"
            )

            # check if face center crosses the line
            # treat line as y = mx + b
            x1_line, y1_line = line_start
            x2_line, y2_line = line_end

            if x2_line - x1_line != 0:  # avoid division by zero
                m = (y2_line - y1_line) / (x2_line - x1_line)
                b = y1_line - m * x1_line
                y_on_line = m * face_center[0] + b

                # if face center is below the line
                if face_center[1] > y_on_line:
                    now = time.time()
                    last_punch_time = get_last_punch_time(person_id)
                    if last_punch_time is None or (now - last_punch_time) > 10:
                        ensure_entry()
                        mark_attendance(person_id, person_name)

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
            0.7,
            (255, 255, 255),
            2,
        )

    return frame


def main():
    global latest_frame, processing, line_start, line_end

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    print("Starting face recognition...")
    print("Press 'q' to quit the program")

    last_process_time = time.time()
    processing_interval = 0.3

    while True:
        current_time = time.time()
        if current_time - last_process_time >= processing_interval:
            with frame_lock:
                if latest_frame is not None and not processing:
                    processing = True
                    frame_to_process = latest_frame.copy()
                    processing = False

                    # Draw the line always
                    cv2.line(frame_to_process, line_start, line_end, (0, 0, 255), 2)

                    # Process the frame
                    processed_frame = process_face(frame_to_process)

                    # Display the frame
                    cv2.imshow("Face Recognition", processed_frame)

                    last_process_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # time.sleep(0.01)

    # capture.release() do later
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
