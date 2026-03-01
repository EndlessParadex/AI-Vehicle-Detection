from ultralytics import YOLO
import cv2, os, csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDER  = "video/Final"
OUTPUT_FOLDER = "video/CNN_test"
CSV_OUTPUT    = "video/cnn_video_evaluation.csv"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PLATE_MODEL_PATH = "runs/plate_1class/yolov8s_6404/weights/best.pt"
CNN_MODEL_PATH   = "cnn_results/best.pt"

CONF_PLATE  = 0.3
CNN_CONF_TH = 0.5

CLASS_NAMES = [
    "plate-ev-black",
    "plate-ev-white",
    "plate-ev-yellow",
    "plate-government",
    "plate-ice-black",
    "plate-ice-white",
    "plate-public",
]

# =====================================================
# LOAD MODELS
# =====================================================
plate_model = YOLO(PLATE_MODEL_PATH).to(DEVICE)

cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, len(CLASS_NAMES))
cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn.to(DEVICE)
cnn.eval()

# =====================================================
# CNN TRANSFORM
# =====================================================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# HELPERS
# =====================================================
def classify_plate(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = cnn(x)
        prob = torch.softmax(out, dim=1)
        conf, cls = prob.max(1)

    if conf.item() < CNN_CONF_TH:
        return None, None

    return CLASS_NAMES[cls.item()], conf.item()

def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - h - 6), (x + w + 6, y), (0, 255, 0), -1)
    cv2.putText(frame, text, (x + 3, y - 3),
                font, scale, (0, 0, 0), thickness)

# =====================================================
# MAIN PIPELINE
# =====================================================
def process_videos():

    stats = defaultdict(lambda: {"count": 0, "conf_sum": 0.0})
    total_predictions = 0

    videos = [v for v in os.listdir(INPUT_FOLDER)
              if v.lower().endswith((".mp4", ".avi", ".mov"))]

    for video in videos:
        print(f"▶ Processing {video}")
        cap = cv2.VideoCapture(os.path.join(INPUT_FOLDER, video))

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))

        writer = cv2.VideoWriter(
            os.path.join(OUTPUT_FOLDER, f"FINAL-{video}"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = plate_model(frame, conf=CONF_PLATE)[0].boxes

            if results is not None:
                for p in results:
                    x1, y1, x2, y2 = map(int, p.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    label, conf = classify_plate(crop)

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 165, 255), 2)

                    if label:
                        draw_label(frame, f"{label} ({conf:.2f})", x1, y1)

                        stats[label]["count"] += 1
                        stats[label]["conf_sum"] += conf
                        total_predictions += 1

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"✔ Saved FINAL-{video}")

    # =====================================================
    # SUMMARY OUTPUT
    # =====================================================
    print("\nCNN RESNET18 VIDEO EVALUATION SUMMARY")
    print("=" * 60)

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Count", "Average_Confidence"])

        for cls, v in stats.items():
            avg_conf = v["conf_sum"] / v["count"] if v["count"] > 0 else 0
            print(f"{cls:20s} | Count: {v['count']:5d} | Avg Conf: {avg_conf:.3f}")
            writer.writerow([cls, v["count"], f"{avg_conf:.4f}"])

    print("=" * 60)
    print(f"Total Valid Predictions : {total_predictions}")
    print(f"Output Video Folder     : {OUTPUT_FOLDER}")
    print(f"CSV Saved               : {CSV_OUTPUT}")
    print("🎉 Semua video selesai")

# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    process_videos()
