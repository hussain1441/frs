# needs video input, will only preview recognition and has no output file

import cv2
import insightface
from pymongo import MongoClient
import numpy as np
import threading
import time
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

# input url
input_file = "input.mp4"

# global variables
latest_frame = None
frame_lock = threading.Lock()
processing = False
reader_done = False

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


def preview_video(input_file):
    global latest_frame, reader_done

    capture = cv2.VideoCapture(input_file)

    # add optional settings to downscale
    # capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # capture.set(cv2.CAP_PROP_FPS, 30)  # try 15 if not working

    if not capture.isOpened():
        print(f"❌ Could not open video: {input_file}")
        reader_done = True
        return

    while True:
        gotFrame, frame = capture.read()

        if not gotFrame:
            print("❌ Did not get frame")
            break

        with frame_lock:
            latest_frame = frame.copy()

    capture.release()
    reader_done = True


def process_face(frame):
    # resize frame

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
        # scale the box

        if best_score > 0.6:
            color = (0, 255, 0)  # green for recognized
            label = (
                f"{best_match['name']} ({best_match['roll_number']}) - {best_score:.2f}"
            )
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
            0.7,
            (255, 255, 255),
            2,
        )

    return frame


def main():
    global latest_frame, processing, input_file

    reader_thread = threading.Thread(
        target=preview_video, args=(input_file,), daemon=True
    )
    reader_thread.start()

    print("Starting face recognition...")
    print("Press 'q' to quit the program")

    last_process_time = time.time()
    processing_interval = 0.03

    while True:
        current_time = time.time()
        if current_time - last_process_time >= processing_interval:
            with frame_lock:
                if latest_frame is not None and not processing:
                    processing = True
                    frame_to_process = latest_frame.copy()
                    processing = False

                    # Process the frame
                    processed_frame = process_face(frame_to_process)

                    # Display the frame
                    cv2.imshow("Face Recognition Preview", processed_frame)

                    last_process_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if reader_done:
            print("✅ Video finished")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
