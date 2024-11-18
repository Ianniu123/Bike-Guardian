import cv2
import numpy as np

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

def detect_bounding_box(frame):
    output = model(frame)
    results = Detections.from_ultralytics(output[0])

    for face in results.xyxy:
        x1 = int(face[0])
        y1 = int(face[1])
        x2 = int(face[2])
        y2 = int(face[3])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if len(results.xyxy > 0):
        return True

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", cache_dir="./models", filename="model.pt")
model = YOLO(model_path)

video_capture = cv2.VideoCapture(0)

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "Bike Gurdian", video_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()