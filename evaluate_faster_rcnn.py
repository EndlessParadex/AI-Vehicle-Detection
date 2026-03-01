import torch
import torchvision
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "faster_rcnn_mobilenetv3_large_320_fpn_vehicle/last.pth"
VAL_IMG_DIR = "dataset_coco/dataset_vehicle_coco/valid/images"
VAL_ANN_FILE = "dataset_coco/dataset_vehicle_coco/valid/annotations.coco.json"

NUM_CLASSES = 6  # 7 class + background + void (sesuai checkpoint)
SCORE_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# DATASET COCO (INFERENCE ONLY)
# =========================================================
class CocoEvalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        return ToTensor()(img), img_id

# =========================================================
# LOAD MODEL
# =========================================================
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=None
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================================================
# LOAD DATA
# =========================================================
dataset = CocoEvalDataset(VAL_IMG_DIR, VAL_ANN_FILE)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================================================
# INFERENCE → FORMAT COCO
# =========================================================
results = []
coco_gt = COCO(VAL_ANN_FILE)
valid_cat_ids = coco_gt.getCatIds()

with torch.no_grad():
    for images, img_ids in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for output, img_id in zip(outputs, img_ids):
            for box, score, label in zip(
                output["boxes"].cpu(),
                output["scores"].cpu(),
                output["labels"].cpu()
            ):
                if score < SCORE_THRESH:
                    continue

                # Pastikan label valid COCO
                if int(label) not in valid_cat_ids:
                    continue

                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })

# =========================================================
# COCO EVALUATION (VALID & RESMI)
# =========================================================
coco_dt = coco_gt.loadRes(results)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# =========================================================
# METRIK RESMI COCO
# =========================================================
map50_95 = coco_eval.stats[0]
map50     = coco_eval.stats[1]
recall    = coco_eval.stats[8]

# Precision aproksimasi COCO
precision = map50

# F1 Score (berbasis precision-recall resmi)
f1_score = (
    2 * precision * recall / (precision + recall + 1e-6)
    if (precision + recall) > 0 else 0
)

# =========================================================
# OUTPUT
# =========================================================
print("\n=== DOCUMENTED MODEL RESULT – FASTER R-CNN (PLATE) ===")
print(f"Precision (approx COCO) : {precision:.4f}")
print(f"Recall (COCO)           : {recall:.4f}")
print(f"F1 Score                : {f1_score:.4f}")
print(f"mAP@50                  : {map50:.4f}")
print(f"mAP@50–95               : {map50_95:.4f}")
