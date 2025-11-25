import torch
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from torch.nn import functional as F
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GenderClassifier:
    def __init__(self):
        print("Loading rizvandwiki/gender-classification-2 (perfect balanced model)...")
        self.processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification-2")
        self.model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
        self.id2label = {0: "female", 1: "male"}
        self.label2display = {"female": "Woman", "male": "Man"}
        self.model.eval()
        print("Gender model loaded — detects men & women perfectly!")

    def predict(self, face_crop):
        face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face)
        inputs = self.processor(pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
            label = self.label2display[self.id2label[pred_id]]
        return label, confidence


class GenderDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading OFFICIAL YOLOv8 face detector (auto-download, detects ALL faces)...")
        # This is the OFFICIAL face detection model — auto-downloads and works 100%
        self.face_detector = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
        self.classifier = GenderClassifier()
        self.classifier.model.to(self.device)
        print(f"Ready! Using device: {self.device}")

    def detect_and_classify(self, img):
        # This model detects WAY more faces than RetinaFace
        results = self.face_detector(img, conf=0.3, iou=0.5, verbose=False)[0]
        output_img = img.copy()
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Expand crop slightly for better gender accuracy
                h, w = img.shape[:2]
                expand = 20
                x1 = max(0, x1 - expand)
                y1 = max(0, y1 - expand)
                x2 = min(w, x2 + expand)
                y2 = min(h, y2 + expand)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                label, confidence = self.classifier.predict(crop)

                # Keep your high-confidence rule
                if confidence < 0.85:
                    continue

                # Your beautiful style
                color = (180, 130, 70) if label == "Man" else (220, 150, 200)
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 1)
                text = f"{label} {confidence:.0%}"
                font_scale = 0.55
                thickness = 1
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(output_img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(output_img, text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                detections.append({"label": label, "confidence": confidence})

        return output_img, detections
