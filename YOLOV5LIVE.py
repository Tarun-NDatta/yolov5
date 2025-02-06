import cv2
import torch
from pathlib import Path


YOLOv5_DIR = Path("yolov5")
MODEL_PATH = YOLOv5_DIR / "yolov5s.pt"  


model = torch.hub.load(
    str(YOLOv5_DIR), 
    'custom', 
    path=str(MODEL_PATH), 
    source='local'
)
model.float()  # Set model to float for better CPU compatibility

# Get the class names dynamically from the model
class_names = model.names
print("Class Names:", class_names)  

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Define the classes to detect (person, bottle, phone, etc.)
target_classes = [0, 13, 24, 39, 63, 67] 


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    
    results = model(frame)
    detections = results.xyxy[0]

    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)

        
        if ((class_id == 0 and conf > 0.6) or 
            (class_id in target_classes[1:] and conf > 0.4)):  
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_names[class_id]} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Live Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
