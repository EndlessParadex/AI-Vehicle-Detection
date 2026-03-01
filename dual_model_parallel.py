from ultralytics import YOLO
import cv2, os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from threading import Thread
from queue import Queue

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDER  = "video/Final"
OUTPUT_FOLDER = "outputs/final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VEHICLE_MODEL_PATH = r"runs/detect/yolov8s_vehicle_optimized/weights/best.pt"
PLATE_MODEL_PATH   = r"runs/plate_1class/yolov8s_6404/weights/best.pt"
CNN_MODEL_PATH     = r"cnn_results/best.pt"

CONF_VEHICLE = 0.2
CONF_PLATE   = 0.2
CNN_CONF_TH  = 0.2

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
vehicle_model = YOLO(VEHICLE_MODEL_PATH).to(DEVICE)
plate_model   = YOLO(PLATE_MODEL_PATH).to(DEVICE)

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
        return None

    return CLASS_NAMES[cls.item()]


def merge_label(vehicle, plate):
    if plate is None:
        return f"{vehicle}-unknown"

    if "government" in plate:
        return f"{vehicle}-government"
    if "public" in plate:
        return f"{vehicle}-public"
    if "ev" in plate:
        return f"{vehicle}-electric"
    if "ice" in plate:
        return f"{vehicle}-combustion"

    return f"{vehicle}-unknown"


def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(frame, (x, y - h - 6), (x + w + 6, y), (0, 255, 0), -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (0, 0, 0), thickness)


def is_inside(px1, py1, px2, py2, vx1, vy1, vx2, vy2):
    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2
    return vx1 <= cx <= vx2 and vy1 <= cy <= vy2


# =====================================================
# THREADING WORKERS
# =====================================================
def detect_plate_thread(frame, queue):
    result = plate_model(frame, conf=CONF_PLATE)[0].boxes
    queue.put(result)


def detect_vehicle_thread(frame, queue):
    result = vehicle_model(frame, conf=CONF_VEHICLE)[0].boxes
    queue.put(result)


# =====================================================
# MAIN PIPELINE (MULTITHREADED)
# =====================================================
def process_videos():
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

            # ===================== START THREADS =====================
            plate_q = Queue()
            vehicle_q = Queue()

            t_plate = Thread(target=detect_plate_thread, args=(frame, plate_q))
            t_vehicle = Thread(target=detect_vehicle_thread, args=(frame, vehicle_q))

            t_plate.start()
            t_vehicle.start()

            t_plate.join()
            t_vehicle.join()

            # ===================== GET RESULTS =====================
            plate_results   = plate_q.get()
            vehicle_results = vehicle_q.get()

            # ===================== PLATES =====================
            plates = []
            for p in plate_results:
                px1, py1, px2, py2 = map(int, p.xyxy[0])
                crop = frame[py1:py2, px1:px2]

                if crop.size == 0:
                    continue

                plate_type = classify_plate(crop)
                plates.append({
                    "bbox": (px1, py1, px2, py2),
                    "type": plate_type
                })

            # ===================== VEHICLES =====================
            vehicles = []
            for v in vehicle_results:
                vx1, vy1, vx2, vy2 = map(int, v.xyxy[0])
                v_label = vehicle_model.names[int(v.cls)].lower()

                vehicles.append({
                    "bbox": (vx1, vy1, vx2, vy2),
                    "label": v_label
                })

            # ===================== MERGE & DRAW =====================
            for plate in plates:
                px1, py1, px2, py2 = plate["bbox"]
                plate_type = plate["type"]

                matched_vehicle = None
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v["bbox"]
                    if is_inside(px1, py1, px2, py2, vx1, vy1, vx2, vy2):
                        matched_vehicle = v
                        break

                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)

                if matched_vehicle:
                    vx1, vy1, vx2, vy2 = matched_vehicle["bbox"]
                    final_label = merge_label(
                        matched_vehicle["label"], plate_type
                    )

                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                    draw_label(frame, final_label.upper(), vx1, vy1)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"✔ Saved FINAL-{video}")

    print("🎉 Semua video selesai")


# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    process_videos()
