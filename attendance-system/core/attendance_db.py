# insertion and deletion of data

from pymongo import MongoClient
from datetime import date, datetime
import time
from dotenv import load_dotenv
import os

uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
collection = db[os.getenv("ATTENDANCE_COLLECTION_NAME")]


def ensure_entry():
    today = str(date.today())
    if attendance_collection.find_one({"date": today}) is None:
        attendance_collection.insert_one({"date": today, "attendance": []})
        print(f"✅ Created attendance entry for {today}")
    else:
        print(f"✅ Attendance entry for {today} already exists")


def get_last_punch_time(person_id: str):
    today = str(date.today())
    doc = attendance_collection.find_one({"date": today})
    if doc is None:
        return None

    for person in doc["attendance"]:
        if person["id"] == person_id:
            return person["lastPunchTime"]

    return None


def mark_attendance(person_id: str, person_name: str):
    today_str = str(date.today())

    # get today's document
    doc = attendance_collection.find_one({"date": today_str})
    if doc is None:
        print("❌ Today's attendance document doesn't exist!")
        return

    # search for the person
    person_record = None
    for person in doc["attendance"]:
        if person["id"] == person_id:
            person_record = person
            break

    # if person not in today's list, create new record
    if person_record is None:
        person_record = {
            "id": person_id,
            "name": person_name,
            "punchInList": [],
            "punchOutList": [],
            "lastPunchTime": None,
        }
        attendance_collection.update_one(
            {"date": today_str}, {"$push": {"attendance": person_record}}
        )

    # determine whether to punch in or punch out
    # logic: if punchInList is empty or lengths equal → punch in, else → punch out
    if len(person_record["punchInList"]) == len(person_record["punchOutList"]):
        # punch in
        attendance_collection.update_one(
            {"date": today_str, "attendance.id": person_id},
            {
                "$push": {"attendance.$.punchInList": datetime.now().isoformat()},
                "$set": {"attendance.$.lastPunchTime": time.time()},
            },
        )
        print(f"☘️ {person_name} punched in at {datetime.now()}")
    else:
        # punch out
        attendance_collection.update_one(
            {"date": today_str, "attendance.id": person_id},
            {
                "$push": {"attendance.$.punchOutList": datetime.now().isoformat()},
                "$set": {"attendance.$.lastPunchTime": time.time()},
            },
        )
        print(f"🏴 {person_name} punched out at {datetime.now()}")


def multi_camera_attendance(person_id: str, person_name: str, camera_name: str):
    today_str = str(date.today())
    doc = attendance_collection.find_one({"date": today_str})
    if doc is None:
        print("❌ Today's attendance document doesn't exist!")
        return

    # find person
    person_record = next((p for p in doc["attendance"] if p["id"] == person_id), None)
    if person_record is None:
        person_record = {
            "id": person_id,
            "name": person_name,
            "punchInList": [],
            "punchOutList": [],
            "lastPunchTime": None,
        }
        attendance_collection.update_one(
            {"date": today_str}, {"$push": {"attendance": person_record}}
        )

    # Decide punch type based on camera
    if camera_name == "PunchIn" and len(person_record["punchInList"]) == len(
        person_record["punchOutList"]
    ):
        attendance_collection.update_one(
            {"date": today_str, "attendance.id": person_id},
            {
                "$push": {"attendance.$.punchInList": datetime.now().isoformat()},
                "$set": {"attendance.$.lastPunchTime": time.time()},
            },
        )
        print(f"☘️ {person_name} punched in via {camera_name} at {datetime.now()}")
    elif camera_name == "PunchOut" and len(person_record["punchInList"]) != len(
        person_record["punchOutList"]
    ):
        attendance_collection.update_one(
            {"date": today_str, "attendance.id": person_id},
            {
                "$push": {"attendance.$.punchOutList": datetime.now().isoformat()},
                "$set": {"attendance.$.lastPunchTime": time.time()},
            },
        )
        print(f"🏴 {person_name} punched out via {camera_name} at {datetime.now()}")
