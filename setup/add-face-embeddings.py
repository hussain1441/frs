import json
import cv2
import insightface
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("EMBEDDING_COLLECTION_NAME")]
print("Connected to MongoDB ✅")

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))

with open("main-shits/all-employees.json", "r") as f:
    users = json.load(f)

for user in users:
    name = user["name"]
    rollno = user["id"]
    path = user["path"]

    img = cv2.imread(path)
    if img is None:
        print(f"❌ error loading image for {name} with roll number {rollno}")
        continue

    print(f"✅ Successfully loaded image {path}")

    faces = model.get(img)  # gives a list of all faces
    face = faces[0]
    if face is None:
        print(f"❌ no face detected for {path}")
        continue

    embedding = face.normed_embedding.astype(float).tolist()  # i embed here

    doc = {
        "roll_number": rollno,
        "name": name,
        "embedding": embedding,
        "created_at": datetime.now(),
    }

    collection.insert_one(doc)
    print(f"💊 Successfully inserted {name} with embedding length {len(embedding)}")

    # cv2.imshow(name, img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# try:
#     result = collection.insert_one({"name": "hussain", "rollno": "22A91A05E3"})

# except Exception as e:
#     print("Error: ", e)

# finally:
#     client.close()
