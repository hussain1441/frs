# check a persons data from the db for a date

from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# MongoDB setup
uri = os.getenv("MONGODB_URL")
client = MongoClient(uri)
db = client[os.getenv("DB_NAME")]
attendance_collection = db[os.getenv("ATTENDANCE_COLLECTION_NAME")]

# Input
date_input = input("Enter date (YYYY-MM-DD): ").strip()
emp_id = input("Enter Employee ID: ").strip()

# Fetch the document for that date
doc = attendance_collection.find_one({"date": date_input})
if not doc:
    print(f"No attendance data for {date_input}")
    exit()

# Find the employee
person = next((p for p in doc["attendance"] if p["id"] == emp_id), None)
if not person:
    print(f"No record found for Employee ID {emp_id} on {date_input}")
    exit()

# Extract punch lists
punch_in_list = person.get("punchInList", [])[:2]  # only first 2
punch_out_list = person.get("punchOutList", [])[:2]  # only first 2

# Convert strings to datetime and format in 12-hour
punch_in_times = [
    datetime.fromisoformat(t).strftime("%I:%M:%S %p") for t in punch_in_list
]
punch_out_times = [
    datetime.fromisoformat(t).strftime("%I:%M:%S %p") for t in punch_out_list
]

first_punch_in = punch_in_times[0] if punch_in_times else None
last_punch_out = punch_out_times[-1] if punch_out_times else None

# Print nicely
print(f"\n📅 Attendance for {person['name']} (ID: {emp_id}) on {date_input}")
print(f"Total Punch-ins: {len(punch_in_times)}")
print(f"Total Punch-outs: {len(punch_out_times)}")

# First punch-in and last punch-out in the same line
print(f"- [First Punch-in]/[Last Punch-out]: [{first_punch_in}]/[{last_punch_out}]")

print(f"- All Punch-ins: {', '.join(punch_in_times)}")
print(f"- All Punch-outs: {', '.join(punch_out_times)}")
