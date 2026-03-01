import os
import torch
from torchvision import models, transforms
from PIL import Image

# =========================
# CONFIG
# =========================
CROP_DIR   = "cnn_raw_crops"          # folder utama hasil crop
MODEL_PATH = "cnn_results/best.pt"    # model CNN terbaik
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRESHOLD = 0.4                  # threshold confidence
STABILITY_K = 5                       # frame berturut-turut

NUM_CLASSES = 7

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# TRANSFORM
# =========================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# METRIC VARIABLES
# =========================
total_frames = 0
detected_frames = 0
confidence_list = []

stable_frames = 0
current_streak = 0

# =========================
# PROCESS ALL VIDEOS
# =========================
for video_folder in sorted(os.listdir(CROP_DIR)):
    video_path = os.path.join(CROP_DIR, video_folder)

    if not os.path.isdir(video_path):
        continue

    print(f"▶ Processing {video_folder}")

    for img_name in sorted(os.listdir(video_path)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(video_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        total_frames += 1

        x = tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            conf, _ = prob.max(1)

        conf_val = conf.item()

        if conf_val >= CONF_THRESHOLD:
            detected_frames += 1
            confidence_list.append(conf_val)

            current_streak += 1
            if current_streak >= STABILITY_K:
                stable_frames += 1
        else:
            current_streak = 0

# =========================
# METRIC CALCULATION
# =========================
DR  = (detected_frames / total_frames) * 100 if total_frames else 0
ACS = sum(confidence_list) / len(confidence_list) if confidence_list else 0
DSR = (stable_frames / detected_frames) * 100 if detected_frames else 0

# =========================
# OUTPUT
# =========================
print("\n" + "=" * 60)
print("CNN VIDEO-BASED EVALUATION RESULT")
print("=" * 60)
print(f"Total Frames Evaluated        : {total_frames}")
print(f"Frames with Detection        : {detected_frames}")
print(f"Detection Rate (DR)          : {DR:.2f}%")
print(f"Average Confidence Score     : {ACS:.3f}")
print(f"Detection Stability Rate     : {DSR:.2f}%")
print("=" * 60)
