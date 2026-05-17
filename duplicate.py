import hashlib
import shutil
from pathlib import Path

DATASET_DIR = Path("merged_gun_knife_dataset_balanced")
DUPLICATE_DIR = Path("duplicates_removed")

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# hash hesaplama (çok önemli)
def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

seen_hashes = set()
duplicate_count = 0

for split in ["train", "valid", "test"]:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    dup_img_dir = DUPLICATE_DIR / split / "images"
    dup_lbl_dir = DUPLICATE_DIR / split / "labels"

    dup_img_dir.mkdir(parents=True, exist_ok=True)
    dup_lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img_hash = get_image_hash(image_path)

        label_path = labels_dir / f"{image_path.stem}.txt"

        if img_hash in seen_hashes:
            # duplicate → taşı
            shutil.move(str(image_path), dup_img_dir / image_path.name)

            if label_path.exists():
                shutil.move(str(label_path), dup_lbl_dir / label_path.name)

            duplicate_count += 1
        else:
            seen_hashes.add(img_hash)

print("Bitti.")
print("Toplam duplicate silinen:", duplicate_count)
print("Taşınan klasör:", DUPLICATE_DIR)