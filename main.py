import cv2
import torch

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from torchvision.transforms import v2
from scipy.spatial.distance import euclidean

read_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    fixed_image_standardization,
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", cache_dir="./models", filename="model.pt")
model = YOLO(model_path)

state = torch.load("./model2.pt", weights_only=True)
verification_model = InceptionResnetV1(classify=False, pretrained='vggface2').to(device)
verification_model.logits = None
verification_model.load_state_dict(state)

original = read_pipeline(Image.open("data\original\Ian\Ian02.jpg")).unsqueeze(0)

def verify_face(filename):
    input = read_pipeline(Image.open(filename)).unsqueeze(0)
    
    verification_model.eval()
    with torch.no_grad():
        emb1 = verification_model(original.to(device)).cpu().numpy().squeeze()
        emb2 = verification_model(input.to(device)).cpu().numpy().squeeze()
        
    distance = euclidean(emb1, emb2)
    print(f"Distance: {distance}")
    
    return distance < 0.7

def detect_bounding_box(frame):
    output = model(frame, verbose=False)
    results = Detections.from_ultralytics(output[0])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    correct_color = (0, 255, 0)
    incorrect_color = (255, 0, 0)
    thickness = 2
    
    for face in results.xyxy:
        x1 = int(face[0])
        y1 = int(face[1])
        x2 = int(face[2])
        y2 = int(face[3])

        cropped_face = frame[y1:y2, x1:x2]
        resized_image = cv2.resize(cropped_face, (160, 160))
        filename = "output.jpg"
        cv2.imwrite(filename, resized_image)
            
        if verify_face(filename):
            cv2.putText(frame, 'Ian', (x1, y1), font, fontScale, correct_color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'UNKNOWN', (x1, y1), font, fontScale, incorrect_color, thickness, cv2.LINE_AA)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                
    if len(results.xyxy > 0):
        return True

video_capture = cv2.VideoCapture(0)

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame,
    )  

    cv2.imshow(
        "Bike Gurdian", video_frame
    )
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()