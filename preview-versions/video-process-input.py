# needs video input, will only process recognition and give output file

import cv2
import insightface
from pymongo import MongoClient
import numpy as np
import threading
import queue
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------
# MongoDB & Model Setup
# ----------------------
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # USE ctx_id=0 FOR GPU

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

# ----------------------
# Config
# ----------------------
input_file = "input.mp4"
output_file = "output.mp4"
show_preview = True

# Queues & threading
frame_queue = queue.Queue(maxsize=16)
processed_queue = queue.Queue(maxsize=16)
reader_done = False


# ----------------------
# Utility Functions
# ----------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def process_face(frame):
    faces = model.get(frame)
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
            color = (0, 255, 0)
            label = (
                f"{best_match['name']} ({best_match['roll_number']}) - {best_score:.2f}"
            )
        else:
            color = (0, 0, 255)
            label = f"Unknown - {best_score:.2f}"

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


# ----------------------
# Thread Functions
# ----------------------
def reader():
    global reader_done
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_file}")
        reader_done = True
        return

    # skip_rate = 0  # process every nth frame
    # frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_counter += 1
        # if frame_counter % skip_rate != 0:
        #     continue  # skip this frame

        frame_queue.put(frame)

    reader_done = True
    cap.release()


def processor():
    while not (reader_done and frame_queue.empty()):
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        processed = process_face(frame)
        processed_queue.put(processed)
        frame_queue.task_done()


# ----------------------
# Main
# ----------------------
def main():
    global reader_done

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_file}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    t_reader = threading.Thread(target=reader, daemon=True)
    t_processor = threading.Thread(target=processor, daemon=True)
    t_reader.start()
    t_processor.start()

    processed_count = 0
    print(f"Processing video ({total_frames} frames) → {output_file}")

    while not (reader_done and processed_queue.empty()):
        try:
            frame = processed_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        writer.write(frame)
        processed_count += 1

        if show_preview:
            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("❌ Stopped by user")
                break

        print(
            f"Progress: {processed_count}/{total_frames} ({processed_count/total_frames*100:.2f}%)",
            end="\r",
        )

    writer.release()
    cv2.destroyAllWindows()
    print("\n✅ Video processing complete!")


if __name__ == "__main__":
    main()
