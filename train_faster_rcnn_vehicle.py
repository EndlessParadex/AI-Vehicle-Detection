import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import datetime

from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#                  CONFIGURATION (VEHICLE)
# ============================================================
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
USE_PATIENCE = False
PATIENCE = 0

# 🔴 SESUAI STRUKTUR FOLDER DI GAMBAR
TRAIN_IMG_DIR = "dataset_coco/dataset_vehicle_coco/train/images"
VAL_IMG_DIR   = "dataset_coco/dataset_vehicle_coco/valid/images"
TRAIN_ANN_FILE = "dataset_coco/dataset_vehicle_coco/train/annotations.coco.json"
VAL_ANN_FILE   = "dataset_coco/dataset_vehicle_coco/valid/annotations.coco.json"

MODEL_DIR = "faster_rcnn_vehicle_mobilenetv3_320_epochs100"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last.pth")
OPTIM_PATH = os.path.join(MODEL_DIR, "optimizer_state.pth")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.txt")

# ============================================================
#                  DATASET COCO
# ============================================================
class RoboFlowDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.transforms = transforms
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_remap = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        while True:
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img = Image.open(os.path.join(self.img_folder, img_info["file_name"])).convert("RGB")

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes, labels = [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_remap[a["category_id"]])

            if len(boxes) == 0:
                idx = (idx + 1) % len(self.ids)
                continue

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id])
            }

            if self.transforms:
                img = self.transforms(img)

            return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
#                  MODEL
# ============================================================
def create_model(num_classes):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ============================================================
#                  EVALUATION
# ============================================================
def evaluate_map(model, loader, coco_gt):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                    if score < 0.3:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label) - 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]

# ============================================================
#                  TRAINING
# ============================================================
def train():
    trainset = RoboFlowDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, ToTensor())
    valset   = RoboFlowDataset(VAL_IMG_DIR, VAL_ANN_FILE, ToTensor())

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    num_classes = len(trainset.cat_ids) + 1
    model = create_model(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    coco_gt = COCO(VAL_ANN_FILE)
    best_map = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        map50_95 = evaluate_map(model, val_loader, coco_gt)
        print(f"Loss: {total_loss:.4f} | mAP50-95: {map50_95:.4f}")

        if map50_95 > best_map:
            best_map = map50_95
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            torch.save(optimizer.state_dict(), OPTIM_PATH)
            print("🔥 Best vehicle model updated!")

        torch.save(model.state_dict(), LAST_MODEL_PATH)

    print("\n=== TRAINING VEHICLE FASTERRCNN SELESAI ===")
    print(f"Model disimpan di folder: {MODEL_DIR}")

if __name__ == "__main__":
    train()
