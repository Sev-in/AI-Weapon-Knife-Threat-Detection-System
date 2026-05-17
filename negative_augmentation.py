import cv2
import random
from pathlib import Path
import albumentations as A

NEGATIVE_DIR = Path("negative_dataset")
OUTPUT_DIR = Path("negative_dataset_augmented")

AUG_PER_IMAGE = 1

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.MotionBlur(blur_limit=5, p=0.25),
    A.GaussNoise(p=0.25),
    A.RandomShadow(p=0.25),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        p=0.3
    ),
])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(NEGATIVE_DIR.glob(f"*{ext}"))

created = 0

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Okunamadı:", image_path)
        continue

    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=image)
        aug_image = augmented["image"]

        new_name = f"{image_path.stem}_neg_aug_{i+1}_{random.randint(1000, 9999)}{image_path.suffix}"
        out_path = OUTPUT_DIR / new_name

        cv2.imwrite(str(out_path), aug_image)
        created += 1

print("Bitti.")
print("Üretilen yeni negatif görsel:", created)
print("Çıktı klasörü:", OUTPUT_DIR)