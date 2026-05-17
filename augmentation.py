import cv2
import random
from pathlib import Path
import albumentations as A

DATASET_DIR = Path("merged_gun_knife_dataset")
SPLIT = "train"

IMAGES_DIR = DATASET_DIR / SPLIT / "images"
LABELS_DIR = DATASET_DIR / SPLIT / "labels"

OUTPUT_IMAGES_DIR = DATASET_DIR / SPLIT / "images"
OUTPUT_LABELS_DIR = DATASET_DIR / SPLIT / "labels"

AUG_PER_IMAGE = 3  # Her görselden 3 yeni görsel üret

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussNoise(p=0.2),
        A.RandomShadow(p=0.2),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            p=0.3
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)

def read_yolo_labels(label_path):
    boxes = []
    class_labels = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            boxes.append([x, y, w, h])
            class_labels.append(class_id)

    return boxes, class_labels


def write_yolo_labels(label_path, boxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for class_id, box in zip(class_labels, boxes):
            x, y, w, h = box

            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            if w <= 0 or h <= 0:
                continue

            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(IMAGES_DIR.glob(f"*{ext}"))

total_created = 0

for image_path in image_paths:
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    boxes, class_labels = read_yolo_labels(label_path)

    if not boxes:
        continue

    for i in range(AUG_PER_IMAGE):
        try:
            augmented = transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )

            aug_image = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            if not aug_boxes:
                continue

            new_name = f"{image_path.stem}_aug_{i+1}_{random.randint(1000, 9999)}"

            out_image_path = OUTPUT_IMAGES_DIR / f"{new_name}{image_path.suffix}"
            out_label_path = OUTPUT_LABELS_DIR / f"{new_name}.txt"

            cv2.imwrite(str(out_image_path), aug_image)
            write_yolo_labels(out_label_path, aug_boxes, aug_labels)

            total_created += 1

        except Exception as e:
            print(f"Hata: {image_path.name} -> {e}")

print(f"Bitti. Oluşturulan yeni görsel sayısı: {total_created}")