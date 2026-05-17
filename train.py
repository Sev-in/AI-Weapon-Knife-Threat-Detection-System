from ultralytics import YOLO

# Dataset yaml yolu (train/val/test + class bilgileri burada)
DATA_YAML = "merged_gun_knife_dataset_balanced/data.yaml"

def main():
    # Önceden eğitilmiş model (small versiyon → hız + doğruluk dengesi)
    model = YOLO("yolov8s.pt")

    model.train(
        # Dataset config
        data=DATA_YAML,

        # Eğitim süresi (dataset kaç kez dönecek)
        epochs=100,

        # Görüntü boyutu (tüm resimler 640x640 yapılır)
        imgsz=640,

        # Aynı anda işlenecek görsel sayısı (VRAM’e göre artırılabilir)
        batch=8,

        # GPU kullanımı (0 = ilk GPU, CPU için "cpu")
        device=0,

        # Çıktı klasörü
        project="runs_detect",

        # Eğitim adı (klasör ismi olacak)
        name="gun_knife_yolov8n",

        # Early stopping (20 epoch gelişme yoksa durur)
        patience=20,

        # Data loading thread sayısı
        workers=2,

        # Dataset RAM’e alınsın mı (RAM azsa False)
        cache=False,

        # Pretrained ağırlık kullan (transfer learning)
        pretrained=True,

        # Optimizer otomatik seçilir (AdamW vs.)
        optimizer="auto",

        # Son 10 epoch'ta mosaic kapatılır (daha gerçek öğrenme)
        close_mosaic=10,

        # ===== AUGMENTATION =====

        # Küçük açı döndürme
        degrees=5,

        # Görsel kaydırma (x,y)
        translate=0.05,

        # Zoom in/out
        scale=0.3,

        # Eğme (çok kullanılmaz)
        shear=0.0,

        # Perspektif değişimi (kapalı tuttuk)
        perspective=0.0,

        # Dikey flip (gerçekçi olmadığı için kapalı)
        flipud=0.0,

        # Yatay flip (çok önemli)
        fliplr=0.5,

        # 4 resmi birleştiren güçlü augment
        mosaic=0.7,

        # İki resmi karıştırır (hafif kullan)
        mixup=0.05,

        # Copy-paste augment (kapalı)
        copy_paste=0.0,

        # Renk augment (kamera/ışık farkı simülasyonu)
        hsv_h=0.015,  # hue
        hsv_s=0.5,    # saturation
        hsv_v=0.3,    # brightness
    )

if __name__ == "__main__":
    main()