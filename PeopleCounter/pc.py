import cv2
import torch
import numpy as np
from torchvision.ops import nms
from ultralytics import YOLO

# Function for soft-NMS
def soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    N = boxes.shape[0]
    for i in range(N):
        tscore = scores[i]
        pos = i + 1

        if i != N-1:
            maxscore = torch.max(scores[pos:])
            maxpos = torch.argmax(scores[pos:]) + pos

        if tscore < maxscore:
            boxes[i, :], boxes[maxpos, :] = boxes[maxpos, :], boxes[i, :]
            scores[i], scores[maxpos] = scores[maxpos], scores[i]
        
        # IOU
        xx1 = torch.maximum(boxes[i, 0], boxes[pos:, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[pos:, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[pos:, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[pos:, 3])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        inter = w * h
        area = (boxes[pos:, 2] - boxes[pos:, 0]) * (boxes[pos:, 3] - boxes[pos:, 1])
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ovr = inter / (area + area_i - inter)

        if method == 1:
            weight = torch.where(ovr > Nt, 1 - ovr, torch.ones(ovr.shape))
        elif method == 2:
            weight = torch.exp(-(ovr * ovr) / sigma)
        
        scores[pos:] = scores[pos:] * weight
        
    keep = scores > threshold
    return boxes[keep], scores[keep]

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Function to count people in a frame
def count_people(frame, model):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extracting bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Extracting scores
    
    # Applying soft-NMS
    boxes, scores = soft_nms(torch.tensor(boxes), torch.tensor(scores))

    # Drawing bounding boxes on the frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Person {scores[i]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame, len(boxes)

# Open video file (use absolute path)
video_path = r'C:\Users\Maishara\Desktop\PeopleCounter\3.mp4'  # Update this to your video file's absolute path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error reading video file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Count people in the frame
    frame, count = count_people(frame, model)
    cv2.putText(frame, f'People Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

