from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs_detect/gun_knife_yolov8n/weights/best.pt")

    metrics = model.val(
        data="merged_gun_knife_dataset_balanced/data.yaml",
        split="test"
    )

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

if __name__ == "__main__":
    main()