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
CSV_OUTPUT    = "video/cnn_video_evaluation_per_video.csv"
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

# =====================================================
# MAIN PIPELINE (PER VIDEO)
# =====================================================
def process_videos():

    videos = sorted([v for v in os.listdir(INPUT_FOLDER)
                     if v.lower().endswith((".mp4", ".avi", ".mov"))])

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video",
            "Total_Detections",
            "Average_Confidence"
        ])

        for video in videos:
            print(f"\n▶ Processing {video}")

            stats = []
            cap = cv2.VideoCapture(os.path.join(INPUT_FOLDER, video))

            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(3)), int(cap.get(4))

            writer_video = cv2.VideoWriter(
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
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2)

                        if label:
                            stats.append(conf)
                            cv2.putText(
                                frame,
                                f"{label} ({conf:.2f})",
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,255,0), 2
                            )

                writer_video.write(frame)

            cap.release()
            writer_video.release()

            total = len(stats)
            avg_conf = sum(stats) / total if total > 0 else 0

            writer.writerow([
                video,
                total,
                f"{avg_conf:.3f}"
            ])

            print(f"✔ {video} | Total: {total} | Avg Conf: {avg_conf:.3f}")

    print("\n🎉 Evaluasi per-video selesai")
    print(f"CSV tersimpan di: {CSV_OUTPUT}")

# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    process_videos()
