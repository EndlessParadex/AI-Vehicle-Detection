from sklearn.model_selection import KFold
from pycocotools.coco import COCO
import json, os
import random

def kfold_split(items, k=5, seed=42):
    random.seed(seed)
    items = items.copy()
    random.shuffle(items)

    fold_size = len(items) // k
    folds = []

    for i in range(k - 1):
        folds.append(items[i * fold_size:(i + 1) * fold_size])

    folds.append(items[(k - 1) * fold_size:])

    return folds

def create_kfold_coco(coco_json, k=5, out_dir="folds"):
    coco = COCO(coco_json)
    img_ids = list(coco.imgs.keys())

    folds = kfold_split(img_ids, k)

    os.makedirs(out_dir, exist_ok=True)

    for fold in range(k):
        val_ids = folds[fold]
        train_ids = [i for f in range(k) if f != fold for i in folds[f]]

        for split, ids in [("train", train_ids), ("val", val_ids)]:
            data = {
                "images": [],
                "annotations": [],
                "categories": coco.dataset["categories"]
            }

            ann_ids = coco.getAnnIds(imgIds=ids)
            data["annotations"] = coco.loadAnns(ann_ids)
            data["images"] = [coco.imgs[i] for i in ids]

            fold_dir = f"{out_dir}/fold_{fold+1}"
            os.makedirs(fold_dir, exist_ok=True)

            with open(f"{fold_dir}/{split}.json", "w") as f:
                json.dump(data, f)

    print("✅ 5-fold COCO split created (no sklearn)")

if __name__ == "__main__":
    create_kfold_coco(
        coco_json="dataset_plate_coco_5fold/annotations_full.json",
        k=5,
        out_dir="dataset_plate_coco_5fold"
    )
