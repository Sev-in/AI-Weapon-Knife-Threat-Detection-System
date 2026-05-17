from ultralytics import YOLO

model = YOLO("runs/detect/runs_detect/gun_knife_yolov8n/weights/best.pt")

# klasördeki tüm görselleri test eder
results = model.predict(
    source="merged_gun_knife_dataset_balanced/test/images",
    conf=0.35,
    save=True
)