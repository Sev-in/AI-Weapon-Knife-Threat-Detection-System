import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from pathlib import Path
import time
import subprocess
import imageio_ffmpeg
import pandas as pd

st.set_page_config(layout="wide")
st.title("AI Weapon & Knife Threat Detection System")

MODEL_PATH = "runs/detect/runs_detect/gun_knife_yolov8n/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

def calculate_threat_score(conf, class_name, box_area_ratio):
    base = conf * 70

    if class_name.lower() == "gun":
        class_bonus = 20
    elif class_name.lower() == "knife":
        class_bonus = 12
    else:
        class_bonus = 5

    size_bonus = min(box_area_ratio * 100, 10)

    score = base + class_bonus + size_bonus
    return min(int(score), 100)

def get_risk_level(score):
    if score >= 70:
        return "HIGH", (0, 0, 255)
    elif score >= 45:
        return "MEDIUM", (0, 165, 255)
    elif score >= 30:
        return "LOW", (0, 255, 255)
    return "SAFE", (0, 255, 0)

def draw_detection(frame, box, label, conf, score, risk, color):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    text = f"{label} {conf:.2f} | Risk: {risk} | Score: {score}"

    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

def get_best_threat(results, frame_shape):
    h, w = frame_shape[:2]
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    best_target = None
    best_score = -1

    for box in boxes:
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = w * h
        box_area_ratio = box_area / frame_area if frame_area > 0 else 0

        score = calculate_threat_score(conf, class_name, box_area_ratio)
        risk, color = get_risk_level(score)

        if score > best_score:
            best_score = score
            best_target = {
                "box": (x1, y1, x2, y2),
                "label": class_name,
                "conf": conf,
                "score": score,
                "risk": risk,
                "color": color
            }

    return best_target

def show_risk_panel(score, risk):
    if risk == "HIGH":
        st.error(f"HIGH RISK DETECTED | Threat Score: {score}/100")
    elif risk == "MEDIUM":
        st.warning(f"MEDIUM RISK | Threat Score: {score}/100")
    elif risk == "LOW":
        st.info(f"LOW RISK | Threat Score: {score}/100")
    else:
        st.success("SAFE")

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Tracking", "Live Tracking"])

# =========================
# IMAGE DETECTION
# =========================
with tab1:
    st.header("Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    image_conf = st.slider("Image Confidence", 0.1, 0.9, 0.45, 0.05)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        results = model.predict(
            image_np,
            conf=image_conf,
            imgsz=640,
            verbose=False
        )

        result_frame = image_np.copy()
        target = get_best_threat(results, result_frame.shape)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, width=450)

        with col2:
            st.subheader("Threat Detection Result")

            if target:
                draw_detection(
                    result_frame,
                    target["box"],
                    target["label"],
                    target["conf"],
                    target["score"],
                    target["risk"],
                    target["color"]
                )

                show_risk_panel(target["score"], target["risk"])

                st.metric("Detected Object", target["label"])
                st.metric("Confidence", f"{target['conf']:.2f}")
                st.metric("Threat Score", f"{target['score']}/100")
                st.metric("Risk Level", target["risk"])
            else:
                show_risk_panel(0, "SAFE")

            st.image(result_frame, width=450)

# =========================
# VIDEO TRACKING
# =========================
with tab2:
    st.header("Video Tracking - Smart Threat Tracking")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    video_conf = st.slider("Video Detection Confidence", 0.1, 0.9, 0.45, 0.05)

    detect_interval = st.number_input(
        "YOLO tekrar kontrol aralığı (frame)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )

    TARGET_WIDTH = 640

    st.info(
        "Video 640 genişliğe küçültülür. YOLO en tehlikeli hedefi seçer, "
        "CSRT tracker takip eder. Belirli aralıklarla YOLO tekrar kontrol eder."
    )

    if uploaded_video:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input.close()

        st.subheader("Original Video")
        left, center, right = st.columns([2, 2, 2])
        with center:
            st.video(temp_input.name)

        if st.button("Detect & Track Video"):
            output_dir = Path("video_results")
            output_dir.mkdir(exist_ok=True)

            timestamp = int(time.time())
            raw_output_path = output_dir / f"tracked_raw_{timestamp}.avi"
            output_path = output_dir / f"tracked_{timestamp}.mp4"

            cap = cv2.VideoCapture(temp_input.name)

            if not cap.isOpened():
                st.error("Video açılamadı.")
                st.stop()

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if original_fps == 0:
                original_fps = 25

            target_width = TARGET_WIDTH
            target_height = int(original_height * (target_width / original_width))

            if target_height % 2 != 0:
                target_height += 1

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                str(raw_output_path),
                fourcc,
                original_fps,
                (target_width, target_height)
            )

            if not writer.isOpened():
                st.error("VideoWriter açılamadı.")
                cap.release()
                st.stop()

            tracker = None
            frame_count = 0
            last_target = None
            logs = []

            progress = st.progress(0)

            high_risk_count = 0
            medium_risk_count = 0
            total_detection_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                frame = cv2.resize(frame, (target_width, target_height))

                need_detection = tracker is None or frame_count % detect_interval == 0

                if tracker is not None and not need_detection:
                    success, bbox = tracker.update(frame)

                    if success and last_target:
                        x, y, w, h = [int(v) for v in bbox]
                        x1, y1, x2, y2 = x, y, x + w, y + h

                        label = last_target["label"]
                        conf = last_target["conf"]

                        box_area_ratio = (w * h) / (target_width * target_height)
                        score = calculate_threat_score(conf, label, box_area_ratio)
                        risk, color = get_risk_level(score)

                        draw_detection(
                            frame,
                            (x1, y1, x2, y2),
                            label,
                            conf,
                            score,
                            risk,
                            color
                        )

                        cv2.putText(
                            frame,
                            "TRACKING MODE",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )
                    else:
                        tracker = None
                        need_detection = True

                if need_detection:
                    results = model.predict(
                        frame,
                        conf=video_conf,
                        imgsz=640,
                        verbose=False
                    )

                    target = get_best_threat(results, frame.shape)

                    if target:
                        x1, y1, x2, y2 = target["box"]

                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                        last_target = target

                        draw_detection(
                            frame,
                            target["box"],
                            target["label"],
                            target["conf"],
                            target["score"],
                            target["risk"],
                            target["color"]
                        )

                        cv2.putText(
                            frame,
                            "YOLO DETECTION MODE",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0),
                            2
                        )

                        total_detection_count += 1

                        if target["risk"] == "HIGH":
                            high_risk_count += 1
                        elif target["risk"] == "MEDIUM":
                            medium_risk_count += 1

                        logs.append({
                            "frame": frame_count,
                            "time_sec": round(frame_count / original_fps, 2),
                            "object": target["label"],
                            "confidence": round(target["conf"], 3),
                            "threat_score": target["score"],
                            "risk_level": target["risk"]
                        })
                    else:
                        tracker = None
                        last_target = None

                        cv2.putText(
                            frame,
                            "SAFE / NO THREAT",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                writer.write(frame)

                if total_frames > 0:
                    progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            writer.release()

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i", str(raw_output_path),
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if not output_path.exists() or output_path.stat().st_size == 0:
                st.error("MP4 çıktı oluşturulamadı.")
                st.stop()

            st.success("Video tracking tamamlandı.")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Detections", total_detection_count)
            col_b.metric("High Risk Events", high_risk_count)
            col_c.metric("Medium Risk Events", medium_risk_count)

            st.subheader("Tracked Video")
            left, center, right = st.columns([2, 2, 2])
            with center:
                st.video(str(output_path))

            if logs:
                st.subheader("Detection Logs")
                st.dataframe(pd.DataFrame(logs), use_container_width=True)

                csv = pd.DataFrame(logs).to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Detection Log CSV",
                    data=csv,
                    file_name="detection_logs.csv",
                    mime="text/csv"
                )

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Tracked Video",
                    data=f,
                    file_name="tracked_video.mp4",
                    mime="video/mp4"
                )

# =========================
# LIVE TRACKING
# =========================
with tab3:
    st.header("Live Tracking - Webcam Smart Threat Tracking")

    live_conf = st.slider("Live Detection Confidence", 0.1, 0.9, 0.45, 0.05)

    live_detect_interval = st.number_input(
        "Canlı YOLO tekrar kontrol aralığı (frame)",
        min_value=5,
        max_value=120,
        value=20,
        step=5
    )

    camera_index = st.number_input(
        "Camera Index",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    start_live = st.button("Start Live Tracking")
    stop_live = st.button("Stop Live Tracking")

    if "live_running" not in st.session_state:
        st.session_state.live_running = False

    if start_live:
        st.session_state.live_running = True

    if stop_live:
        st.session_state.live_running = False

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    logs_placeholder = st.empty()

    if st.session_state.live_running:
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error("Kamera açılamadı.")
            st.session_state.live_running = False
        else:
            tracker = None
            frame_count = 0
            last_target = None

            total_detection_count = 0
            high_risk_count = 0
            medium_risk_count = 0

            live_logs = []

            while st.session_state.live_running:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1
                frame = cv2.resize(frame, (640, 480))

                need_detection = tracker is None or frame_count % live_detect_interval == 0

                if tracker is not None and not need_detection:
                    success, bbox = tracker.update(frame)

                    if success and last_target:
                        x, y, w, h = [int(v) for v in bbox]
                        x1, y1, x2, y2 = x, y, x + w, y + h

                        label = last_target["label"]
                        conf = last_target["conf"]

                        box_area_ratio = (w * h) / (640 * 480)
                        score = calculate_threat_score(conf, label, box_area_ratio)
                        risk, color = get_risk_level(score)

                        draw_detection(frame, (x1, y1, x2, y2), label, conf, score, risk, color)

                        with status_placeholder:
                            show_risk_panel(score, risk)

                        # 🔥 LOG EKLE (tracking sırasında)
                        live_logs.append({
                            "frame": frame_count,
                            "time": time.strftime("%H:%M:%S"),
                            "object": label,
                            "confidence": round(conf, 3),
                            "threat_score": score,
                            "risk_level": risk,
                            "mode": "tracking"
                        })

                    else:
                        tracker = None
                        need_detection = True

                if need_detection:
                    results = model.predict(
                        frame,
                        conf=live_conf,
                        imgsz=640,
                        verbose=False
                    )

                    target = get_best_threat(results, frame.shape)

                    if target:
                        x1, y1, x2, y2 = target["box"]

                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                        last_target = target

                        draw_detection(
                            frame,
                            target["box"],
                            target["label"],
                            target["conf"],
                            target["score"],
                            target["risk"],
                            target["color"]
                        )

                        total_detection_count += 1

                        if target["risk"] == "HIGH":
                            high_risk_count += 1
                        elif target["risk"] == "MEDIUM":
                            medium_risk_count += 1

                        with status_placeholder:
                            show_risk_panel(target["score"], target["risk"])

                        # 🔥 LOG EKLE (YOLO detection)
                        live_logs.append({
                            "frame": frame_count,
                            "time": time.strftime("%H:%M:%S"),
                            "object": target["label"],
                            "confidence": round(target["conf"], 3),
                            "threat_score": target["score"],
                            "risk_level": target["risk"],
                            "mode": "yolo"
                        })

                    else:
                        tracker = None
                        last_target = None

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(frame_rgb, width=640)

                with metrics_placeholder:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Live Detections", total_detection_count)
                    col2.metric("High Risk", high_risk_count)
                    col3.metric("Medium Risk", medium_risk_count)

                # 🔥 SON 20 LOG GÖSTER
                if live_logs:
                    logs_placeholder.dataframe(
                        pd.DataFrame(live_logs[-20:]),
                        use_container_width=True
                    )

            cap.release()

            # 🔥 CANLI BİTTİKTEN SONRA FULL LOG
            if live_logs:
                df = pd.DataFrame(live_logs)

                st.subheader("Live Detection Logs")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Live Logs CSV",
                    data=csv,
                    file_name="live_logs.csv",
                    mime="text/csv"
                )