import os
from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "artifacts/yolov8_face_mask/weights/best.pt"
IMAGE_DIR = "data/processed/images/test"
OUTPUT_DIR = "artifacts/inference_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

image_paths = list(Path(IMAGE_DIR).glob("*.png")) + list(Path(IMAGE_DIR).glob("*.jpg"))

print(f"Found {len(image_paths)} test images")

for img_path in image_paths[:10]:  # limit to 10 images
    results = model(img_path, conf=0.25)

    annotated_img = results[0].plot()

    output_path = os.path.join(OUTPUT_DIR, img_path.name)
    cv2.imwrite(output_path, annotated_img)

    print(f"Saved: {output_path}")

print("Inference completed.")
