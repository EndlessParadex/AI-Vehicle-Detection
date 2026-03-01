import torch
import torchvision
import cv2
from ultralytics import YOLO
import numpy as np

# ---------------------------------------------------------
# 1. LOAD MODELS
# ---------------------------------------------------------

# YOLO kendaraan
vehicle_model = YOLO(r"runs/detect/Yolov8-AdamW-150-vehicle/weights/best.pt")

# FasterRCNN plat nomor
faster_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
num_classes = 9  # background + 8 kelas dataset
in_features = faster_model.roi_heads.box_predictor.cls_score.in_features
faster_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

faster_model.load_state_dict(torch.load(
    r"faster_rcnn_mobilenetv3_large_320_fpn/best.pth",
    map_location="cpu"
))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_model = faster_model.to(device)
faster_model.eval()

# ---------------------------------------------------------
# 2. LABEL DAN RULE ENGINE
# ---------------------------------------------------------

PLATE_NAMES = [
    "background",
    "vehicle-plate",
    "plate-ev-black",
    "plate-ev-white",
    "plate-ev-yellow",
    "plate-goverment",
    "plate-ice-black",
    "plate-ice-white",
    "plate-public",
]

PLATE_GROUP = {
    "plate-ev-black": ("ev", "black"),
    "plate-ev-white": ("ev", "white"),
    "plate-ev-yellow": ("ev", "yellow"),

    "plate-goverment": ("gov", "red"),

    "plate-ice-black": ("ice", "black"),
    "plate-ice-white": ("ice", "white"),

    "plate-public": ("public", "yellow"),
}

def classify(vehicle_type: str, plate_label: str | None) -> str:
    vehicle_type = vehicle_type.lower()

    if plate_label is None or plate_label not in PLATE_GROUP:
        return f"{vehicle_type}-unknown"

    plate_category, plate_color = PLATE_GROUP[plate_label]

    if plate_category == "gov":
        return f"{vehicle_type}-government"

    if plate_category == "public":
        return f"{vehicle_type}-public"

    if plate_category == "ev":
        if plate_color in ("black", "white"):
            return f"{vehicle_type}-electric"
        if plate_color == "yellow":
            if vehicle_type in ("car", "motorcycle"):
                return f"{vehicle_type}-electric"
            return f"{vehicle_type}-public"

    if plate_category == "ice":
        return f"{vehicle_type}-combustion"

    return f"{vehicle_type}-unknown"


# Warna kendaraan
colors = {
    "car": (255, 0, 0),
    "motorcycle": (0, 0, 255),
    "bus": (0, 255, 255),
    "truck": (255, 255, 0),
}

# ---------------------------------------------------------
# 3. DRAW LABEL
# ---------------------------------------------------------

def draw_label(frame, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (0, 0, 0), thickness)


# ---------------------------------------------------------
# 4. FASTER-RCNN PLATE DETECTOR
# ---------------------------------------------------------

@torch.no_grad()
def detect_plate(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None, None

    img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    outputs = faster_model(img)[0]

    if len(outputs["scores"]) == 0:
        return None, None

    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    boxes  = outputs["boxes"].cpu().numpy()

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score < 0.5:
        return None, None

    best_label_idx = labels[best_idx]

    if best_label_idx == 0:  # background
        return None, None

    best_box = boxes[best_idx].astype(int)
    best_label = PLATE_NAMES[best_label_idx]

    return best_box, best_label


# ---------------------------------------------------------
# 5. MAIN LOOP
# ---------------------------------------------------------

cap = cv2.VideoCapture(r"video\Batch_2\VID_20251208_143321.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = vehicle_model(frame)[0]

    for v in results.boxes:
        x1, y1, x2, y2 = map(int, v.xyxy[0])
        cls_id = int(v.cls)
        vehicle_label = vehicle_model.names[cls_id]
        v_conf = float(v.conf[0])

        # ROI plat nomor
        vh = y2 - y1
        plate_y1 = int(y1 + vh * 0.55)
        plate_y1 = max(0, min(plate_y1, frame.shape[0] - 1))

        roi = frame[plate_y1:y2, x1:x2]

        pbox, plabel = detect_plate(roi)

        # Vehicle box
        box_color = colors.get(vehicle_label.lower(), (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Plate box
        if pbox is not None:
            px1, py1, px2, py2 = pbox
            px1 += x1
            px2 += x1
            py1 += plate_y1
            py2 += plate_y1
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)

        # Final category
        final_category = classify(
            vehicle_label.lower(),
            plabel.lower() if plabel else None
        )

        final_text = f"{final_category} ({v_conf:.2f})"
        draw_label(frame, final_text, x1, y1, (0, 255, 0))

    cv2.imshow("Combined Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
