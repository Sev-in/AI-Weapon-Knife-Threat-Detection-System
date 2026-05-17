# 🔫 AI Weapon & Knife Threat Detection System

Bu proje, gerçek zamanlı olarak silah ve bıçak tespiti yapabilen, tehdit seviyesini analiz eden ve video/kamera üzerinde takip gerçekleştiren yapay zeka tabanlı bir güvenlik sistemidir.

---

## 🚀 Özellikler

- 🔍 YOLOv8 tabanlı nesne tespiti (Gun & Knife)
- 🎯 Akıllı hedef seçimi (en tehlikeli nesne)
- 📊 Threat Score (0–100 arası risk puanı)
- 🚨 Risk Level analizi (SAFE / LOW / MEDIUM / HIGH)
- 🎥 Video üzerinde tracking (YOLO + CSRT)
- 📹 Canlı kamera (webcam) takibi
- 📁 Detection log sistemi (CSV export)
- ⚡ Optimize edilmiş video pipeline (640px resize)

---

## 🧠 Sistem Mantığı

1. YOLO modeli nesneleri tespit eder
2. En yüksek riskli nesne seçilir
3. Threat Score hesaplanır
4. CSRT tracker ile takip edilir
5. Belirli aralıklarla yeniden detection yapılır

---

## ⚙️ Kurulum

```bash
git clone https://github.com/KULLANICI_ADI/REPO_ADI.git
cd REPO_ADI

pip install ultralytics streamlit opencv-contrib-python numpy pillow pandas imageio-ffmpeg
```

---

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

---

## 🖼️ Kullanım

### 1. Image Detection

- Görsel yükle
- Model tehdit analizi yapar

### 2. Video Tracking

- Video yükle
- Sistem:
  - Silahı tespit eder
  - Takip eder
  - Log oluşturur

### 3. Live Tracking

- Webcam açılır
- Gerçek zamanlı:
  - Detection
  - Tracking
  - Risk analizi yapılır

---

## 📊 Threat Score Sistemi

Threat Score şu faktörlere göre hesaplanır:

- Model confidence
- Nesne türü (gun > knife)
- Nesne boyutu (yakınlık)

| Score  | Risk Level |
| ------ | ---------- |
| 0–29   | SAFE       |
| 30–44  | LOW        |
| 45–69  | MEDIUM     |
| 70–100 | HIGH       |

---

## 📁 Proje Yapısı

```
project/
│
├── app.py
├── train.py
├── val.py
├── predict.py
│
├── runs/
│   └── detect/
│       └── runs_detect/
│           └── gun_knife_yolov8n/
│               └── weights/
│                   └── best.pt
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🤖 Model

Bu projede kullanılan model:

```
runs/detect/runs_detect/gun_knife_yolov8n/weights/best.pt
```

YOLOv8 ile eğitilmiştir.

---

## 📊 Model Performance

Model evaluation results:

- mAP@50: **0.98**
- mAP@50-95: **0.81**
- Precision: **0.97**
- Recall: **0.96**

---

## 📈 Logging Sistemi

- Frame bazlı kayıt
- Threat score
- Risk level
- CSV export

---

## 🚀 Geliştirme Fikirleri

- 🔊 Sesli alarm sistemi
- 📡 Servo + lazer entegrasyonu
- 🧠 Multi-model doğrulama (false positive azaltma)
- 📊 Risk zaman grafiği

---

## 👨‍💻 Geliştirici

Bu proje yapay zeka tabanlı güvenlik sistemleri üzerine geliştirilmiştir.

Şevin Sönmez

Hatice Betül Aydemir

---
