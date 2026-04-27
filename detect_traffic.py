from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("traffic.mp4")

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    vehicle_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls in [2,3,5,7]:
                vehicle_count += 1

    print("Vehicles detected:", vehicle_count)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()