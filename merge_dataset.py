import shutil
from pathlib import Path

NEGATIVE_DIR = Path("negative_dataset_augmented")
TARGET_DATASET = Path("merged_gun_knife_dataset")

TRAIN_IMAGES = TARGET_DATASET / "train" / "images"
TRAIN_LABELS = TARGET_DATASET / "train" / "labels"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

count = 0

for image_path in NEGATIVE_DIR.iterdir():
    if image_path.suffix.lower() not in IMAGE_EXTS:
        continue

    new_name = f"neg_{count:06d}{image_path.suffix}"

    target_image = TRAIN_IMAGES / new_name
    target_label = TRAIN_LABELS / f"{new_name.split('.')[0]}.txt"

    shutil.copy2(image_path, target_image)

    # boş label oluştur
    with open(target_label, "w") as f:
        pass

    count += 1

print("Bitti.")
print("Eklenen negatif:", count)