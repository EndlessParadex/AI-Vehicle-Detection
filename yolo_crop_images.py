from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# =====================
# CONFIG
# =====================
MODEL_PATH = r"runs/plate_1class/yolov8s_6404/weights/best.pt"

VIDEO_DIR = "video/Videos"
OUT_VIDEO_DIR = "outputs/Videos"
OUT_CROP_DIR  = "outputs/crops"

CONF_THRES = 0.5
SAVE_CROP = True   # ubah ke False kalau tidak mau crop

os.makedirs(OUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_CROP_DIR, exist_ok=True)

# =====================
# LOAD MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# PROCESS ALL VIDEOS
# =====================
video_files = list(Path(VIDEO_DIR).glob("*.*"))

print(f"🎥 Found {len(video_files)} videos")

for video_path in video_files:
    print(f"\n▶ Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(
        OUT_VIDEO_DIR,
        f"{video_path.stem}_detected.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    # folder crop per video
    crop_dir = os.path.join(OUT_CROP_DIR, video_path.stem)
    if SAVE_CROP:
        os.makedirs(crop_dir, exist_ok=True)

    frame_idx = 0
    crop_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRES, verbose=False)
        annotated = results[0].plot()

        # save video frame
        out.write(annotated)

        # save crop plate
        if SAVE_CROP:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_path = os.path.join(
                    crop_dir,
                    f"plate_{crop_idx:06d}.jpg"
                )
                cv2.imwrite(crop_path, crop)
                crop_idx += 1

        frame_idx += 1

    cap.release()
    out.release()

    print(f"✔ Saved video : {out_video_path}")
    if SAVE_CROP:
        print(f"✔ Saved crops : {crop_idx} images")

print("\n✅ ALL VIDEOS PROCESSED")
