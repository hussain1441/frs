# needs image input, only preview (no ouput file)

import cv2
import insightface
from pymongo import MongoClient
import numpy as np
import threading
import time
from dotenv import load_dotenv
import os

load_dotenv()

# image url
image_url = "test.png"

# mongodb connection
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

# loading model
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))  # USE ctx_id=0 FOR GPU

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
            1.5,
            (255, 255, 255),
            3,
        )

    return frame


def main():
    global image_url
    frame = cv2.imread(image_url)

    if frame is None:
        print("❌ could not get the image")
        return

    processed_frame = process_face(frame)
    print("Press any key to quit")

    cv2.imshow("Face Recognition", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
