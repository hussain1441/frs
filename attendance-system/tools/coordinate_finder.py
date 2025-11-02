# to figure out start and end of the line

import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")


# cap = cv2.VideoCapture("rtsp://localhost:8554/mystream")  # or your file
cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.2:8554/live")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
