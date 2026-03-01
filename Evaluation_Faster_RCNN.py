import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import torchvision.transforms as T

# =========================
# CONFIG
# =========================
VIDEO_PATH = "video/Phone/Video_4_Phone.mp4"
MODEL_PATH = "faster_rcnn_mobilenetv3_large_320_fpn_plate_1class/best.pth"
OUTPUT_VIDEO = "output/video_eval_fasterrcnn_plate_10.mp4"

CONF_THRESHOLD = 0.4
STABILITY_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL (NUM_CLASSES HARUS SAMA)
# =========================
NUM_CLASSES = 3   # ← WAJIB SAMA DENGAN TRAINING

model = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = T.Compose([T.ToTensor()])

# =========================
# VIDEO LOAD
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# =========================
# METRICS
# =========================
total_frames = 0
detected_frames = 0
confidence_list = []
stable_frames = 0
current_streak = 0
valid_boxes_total = 0

# =========================
# PROCESS VIDEO
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    img = transform(frame).to(DEVICE)

    with torch.no_grad():
        output = model([img])[0]

    scores = output["scores"].cpu().numpy()
    boxes  = output["boxes"].cpu().numpy()

    valid_scores = scores[scores >= CONF_THRESHOLD]

    if len(valid_scores) > 0:
        detected_frames += 1
        valid_boxes_total += len(valid_scores)

        max_conf = valid_scores.max()
        confidence_list.append(max_conf)

        current_streak += 1
        if current_streak >= STABILITY_K:
            stable_frames += 1

        for box, score in zip(boxes, scores):
            if score < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    else:
        current_streak = 0

    writer.write(frame)

cap.release()
writer.release()

# =========================
# METRIC CALCULATION
# =========================
DR  = (detected_frames / total_frames) * 100
ACS = sum(confidence_list) / len(confidence_list) if confidence_list else 0
DSR = (stable_frames / detected_frames) * 100 if detected_frames else 0

print("=== FASTER R-CNN VIDEO EVALUATION ===")
print(f"Detection Rate (DR)  : {DR:.2f}%")
print(f"Avg Confidence (ACS) : {ACS:.3f}")
print(f"Stability (DSR)      : {DSR:.2f}%")
print(f"Output video saved  : {OUTPUT_VIDEO}")
