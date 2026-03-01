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
#                  CONFIGURATION
# ============================================================
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
USE_PATIENCE = False
PATIENCE = 15

TRAIN_IMG_DIR = "dataset_coco/dataset_plate_coco_1class/train/images"
VAL_IMG_DIR = "dataset_coco/dataset_plate_coco_1class/valid/images"
TRAIN_ANN_FILE = "dataset_coco/dataset_plate_coco_1class/train/_annotations.coco.json"
VAL_ANN_FILE = "dataset_coco/dataset_plate_coco_1class/valid/_annotations.coco.json"

MODEL_DIR = "faster_rcnn_mobilenetv3_large_320_fpn"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last.pth")
OPTIM_PATH = os.path.join(MODEL_DIR, "optimizer_state.pth")
LOG_FILE = os.path.join(MODEL_DIR, "training_log.txt")


# ============================================================
#                  DATASET COCO CUSTOM
# ============================================================
class RoboFlowDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.transforms = transforms
        self.img_folder = img_folder

        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_remap = {cid: i+1 for i, cid in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        while True:
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info["file_name"]

            img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            boxes, labels = [], []

            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:  
                    continue  # skip invalid boxes

                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_remap[a["category_id"]])

            # === FIX PALING PENTING ===
            if len(boxes) == 0:
                # pilih index lain (naik 1)
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
#                  PRINT CLASSES
# ============================================================
def print_coco_categories(file):
    c = COCO(file)
    cats = c.loadCats(c.getCatIds())
    print("\nDataset Categories:", [x["name"] for x in cats])


# ============================================================
#                  MODEL CREATION
# ============================================================
def create_model(num_classes):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,      # positional arg di TorchVision lama
        num_classes
    )

    return model


# ============================================================
#                  EVALUATION (mAP, P/R/F1)
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

                for box, score, label in zip(output["boxes"].cpu(),
                                             output["scores"].cpu(),
                                             output["labels"].cpu()):
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

    # compute custom precision/recall
    tp = sum([np.sum(e["dtMatches"][0] > 0)
              for e in coco_eval.evalImgs if e])
    fp = sum([np.sum(e["dtMatches"][0] == 0)
              for e in coco_eval.evalImgs if e])
    fn = sum([np.sum(e["gtMatches"][0] == 0)
              for e in coco_eval.evalImgs if e])

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return coco_eval.stats[0], precision, recall, f1, coco_eval


# ============================================================
#                  TRAINING LOOP
# ============================================================
def train():
    # Init Log
    f = open(LOG_FILE, "w")
    f.write(f"Training Log - {datetime.datetime.now()}\n")

    print_coco_categories(TRAIN_ANN_FILE)
    print_coco_categories(VAL_ANN_FILE)

    trainset = RoboFlowDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, ToTensor())
    valset = RoboFlowDataset(VAL_IMG_DIR, VAL_ANN_FILE, ToTensor())

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    num_classes = len(trainset.cat_ids) + 1

    model = create_model(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_map = 0
    patience_counter = 0

    # Resume if exist
    if os.path.exists(LAST_MODEL_PATH):
        print("Loading previous checkpoint...")
        model.load_state_dict(torch.load(LAST_MODEL_PATH))
        optimizer.load_state_dict(torch.load(OPTIM_PATH))

    coco_gt = COCO(VAL_ANN_FILE)
    class_names = [x["name"] for x in coco_gt.loadCats(coco_gt.getCatIds())]

    train_losses, val_maps = [], []

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()

        total_loss = 0

        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # VALIDATION
        map50_95, prec, rec, f1, coco_eval = evaluate_map(model, val_loader, coco_gt)
        val_maps.append(map50_95)

        print(f"Loss: {avg_loss:.4f} | mAP50-95: {map50_95:.4f}")

        # SAVE BEST
        if map50_95 > best_map:
            best_map = map50_95
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            torch.save(optimizer.state_dict(), OPTIM_PATH)
            print("🔥 Best model updated!")
            patience_counter = 0
        else:
            patience_counter += 1

        # SAVE LAST
        torch.save(model.state_dict(), LAST_MODEL_PATH)
        torch.save(optimizer.state_dict(), OPTIM_PATH)

        if USE_PATIENCE and patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    f.close()

    # PLOT
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(val_maps)
    plt.title("mAP Curve")
    plt.savefig("map_curve.png")


if __name__ == "__main__":
    train()
