import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = r"runs/plate_1class/yolov8s_6404/weights/best.pt"
CNN_MODEL_PATH  = r"cnn_results/best.pt"

VIDEO_DIR = "video/Phone"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5

CLASS_NAMES = [
    "plate-ev-black",
    "plate-ev-white",
    "plate-ev-yellow",
    "plate-government",
    "plate-ice-black",
    "plate-ice-white",
    "plate-public",
]

CLASS_COLORS = {
    "plate-ev-black":   (0, 0, 0),
    "plate-ev-white":   (220, 220, 220),
    "plate-ev-yellow":  (0, 255, 255),
    "plate-government": (0, 0, 255),
    "plate-ice-black":  (60, 60, 60),
    "plate-ice-white":  (245, 245, 245),
    "plate-public":     (255, 0, 0),
}

# =========================
# UTIL: DRAW LABEL
# =========================
def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(
        img,
        (x, y - th - 8),
        (x + tw + 6, y),
        color,
        -1
    )

    text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)

    cv2.putText(
        img,
        text,
        (x + 3, y - 4),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

# =========================
# LOAD MODELS
# =========================
yolo = YOLO(YOLO_MODEL_PATH)

cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, len(CLASS_NAMES))
cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn.to(DEVICE)
cnn.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# PROCESS VIDEOS
# =========================
for video_name in os.listdir(VIDEO_DIR):
    if not video_name.lower().endswith((".mp4", ".mov", ".avi")):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    out_path   = os.path.join(OUT_DIR, video_name.replace(".", "_out."))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    print(f"▶ Processing: {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, conf=CONF_THRES)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                x = tf(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = cnn(x)
                    prob = torch.softmax(out, dim=1)
                    conf, cls = prob.max(1)

                class_name = CLASS_NAMES[cls.item()]
                score = conf.item()
                color = CLASS_COLORS[class_name]

                label = f"{class_name} {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, label, x1, y1, color)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✔ Saved: {out_path}")

print("🎉 Semua video selesai diproses")
