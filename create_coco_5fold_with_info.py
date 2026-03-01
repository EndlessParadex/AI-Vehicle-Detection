import json
import os
import random
from pycocotools.coco import COCO

# ============================================================
# CONFIG
# ============================================================
COCO_JSON = "dataset_plate_coco_5fold/annotations_full.json"
OUT_DIR   = "dataset_plate_coco_5fold"
K         = 5
SEED      = 42

random.seed(SEED)

# ============================================================
# LOAD COCO
# ============================================================
coco = COCO(COCO_JSON)
img_ids = list(coco.imgs.keys())
random.shuffle(img_ids)

fold_size = len(img_ids) // K
folds = []

for i in range(K - 1):
    folds.append(img_ids[i * fold_size:(i + 1) * fold_size])

folds.append(img_ids[(K - 1) * fold_size:])

# ============================================================
# SHARED METADATA (WAJIB UNTUK COCOeval)
# ============================================================
INFO = {
    "description": "5-Fold COCO Dataset - Plate Detection",
    "version": "1.0",
    "year": 2025,
    "contributor": "Endless",
    "date_created": "2025-01-01"
}

LICENSES = [{
    "id": 1,
    "name": "Unknown",
    "url": ""
}]

# ============================================================
# CREATE FOLDS
# ============================================================
for fold_idx in range(K):
    val_ids = folds[fold_idx]
    train_ids = [i for f in range(K) if f != fold_idx for i in folds[f]]

    fold_dir = os.path.join(OUT_DIR, f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        data = {
            "info": INFO,
            "licenses": LICENSES,
            "images": [],
            "annotations": [],
            "categories": coco.dataset["categories"]
        }

        ann_ids = coco.getAnnIds(imgIds=ids)
        data["annotations"] = coco.loadAnns(ann_ids)
        data["images"] = [coco.imgs[i] for i in ids]

        out_path = os.path.join(fold_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(data, f)

        print(f"✔ Created {out_path} | images={len(data['images'])}")

print("\n✅ All 5 folds created with valid COCO metadata.")
