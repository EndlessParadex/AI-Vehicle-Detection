import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision
from torchvision.transforms import ToTensor
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "dataset_plate_coco_5fold"
IMG_ROOT  = os.path.join(DATA_ROOT, "images")

NUM_FOLDS  = 5
BATCH_SIZE = 8
EPOCHS     = 150
LR          = 1e-4

OUT_DIR = "runs_fasterrcnn_5fold"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
class CocoDataset(Dataset):
    def __init__(self, img_root, ann_file):
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_map = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}

        self.transform = ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_root, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []

        for a in anns:
            x, y, w, h = a["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_map[a["category_id"]])

        # skip image tanpa bbox
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.ids))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================================
# MODEL
# ============================================================
def build_model(num_classes):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model

# ============================================================
# EVALUATION (COCO mAP)
# ============================================================
def evaluate_map(model, loader, coco_gt):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for t, o in zip(targets, outputs):
                image_id = int(t["image_id"])
                for box, score, label in zip(
                    o["boxes"], o["scores"], o["labels"]
                ):
                    if score < 0.3:
                        continue
                    x1, y1, x2, y2 = box.cpu().tolist()
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label) - 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]

# ============================================================
# 5-FOLD TRAINING
# ============================================================
def run_5fold():
    fold_maps = []

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n================ FOLD {fold} =================")

        train_json = f"{DATA_ROOT}/fold_{fold}/train.json"
        val_json   = f"{DATA_ROOT}/fold_{fold}/val.json"

        trainset = CocoDataset(IMG_ROOT, train_json)
        valset   = CocoDataset(IMG_ROOT, val_json)

        train_loader = DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            valset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        num_classes = len(trainset.cat_ids) + 1
        model = build_model(num_classes).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        coco_gt = COCO(val_json)

        best_map = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for images, targets in tqdm(
                train_loader,
                desc=f"Fold {fold} | Epoch {epoch+1}"
            ):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mAP = evaluate_map(model, val_loader, coco_gt)

            print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | mAP {mAP:.4f}")

            if mAP > best_map:
                best_map = mAP
                torch.save(
                    model.state_dict(),
                    f"{OUT_DIR}/fold_{fold}_best.pth"
                )
                print("🔥 Best model updated")

        fold_maps.append(best_map)

    print("\n================ FINAL RESULT =================")
    print(f"Mean mAP : {np.mean(fold_maps):.4f}")
    print(f"Std  mAP : {np.std(fold_maps):.4f}")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_5fold()
