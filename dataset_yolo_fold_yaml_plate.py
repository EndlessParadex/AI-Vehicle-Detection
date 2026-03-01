import os
import random
from glob import glob
import yaml

SOURCE_YAML = "data_plate.yaml"
DATASET_ROOT = "dataset_plate_640"
OUTPUT_ROOT = "dataset_plate_5fold"

random.seed(42)

# Load class info
with open(SOURCE_YAML, "r") as f:
    base_yaml = yaml.safe_load(f)

names = base_yaml["names"]
nc = base_yaml["nc"]

# Gather all images from train + valid + test
all_images = []
for sub in ["train/images", "valid/images", "test/images"]:
    p = os.path.join(DATASET_ROOT, sub)
    if os.path.exists(p):
        all_images.extend(glob(p + "/*.jpg"))

print(f"Total images ditemukan: {len(all_images)}")

random.shuffle(all_images)

fold_size = len(all_images) // 5
folds = [all_images[i * fold_size:(i + 1) * fold_size] for i in range(5)]
remainder = len(all_images) - fold_size * 5
if remainder > 0:
    folds[-1] += all_images[-remainder:]

def hardlink(src, dst):
    if not os.path.exists(dst):
        os.link(src, dst)

for i in range(5):
    fold_id = i + 1

    fold_dir = os.path.join(OUTPUT_ROOT, f"fold{fold_id}")
    train_img = os.path.join(fold_dir, "train/images")
    train_lbl = os.path.join(fold_dir, "train/labels")
    val_img   = os.path.join(fold_dir, "val/images")
    val_lbl   = os.path.join(fold_dir, "val/labels")

    os.makedirs(train_img, exist_ok=True)
    os.makedirs(train_lbl, exist_ok=True)
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lbl, exist_ok=True)

    val_set = folds[i]
    train_set = [x for j,f in enumerate(folds) if j != i for x in f]

    print(f"Fold {fold_id}: train={len(train_set)}, val={len(val_set)}")

    # TRAIN hardlink
    for img in train_set:
        lbl = img.replace("images", "labels").replace(".jpg", ".txt")

        hardlink(img, os.path.join(train_img, os.path.basename(img)))
        hardlink(lbl, os.path.join(train_lbl, os.path.basename(lbl)))

    # VAL hardlink
    for img in val_set:
        lbl = img.replace("images", "labels").replace(".jpg", ".txt")

        hardlink(img, os.path.join(val_img, os.path.basename(img)))
        hardlink(lbl, os.path.join(val_lbl, os.path.basename(lbl)))

    # Generate YAML
    yaml_data = {
        "train": f"{OUTPUT_ROOT}/fold{fold_id}/train/images",
        "val": f"{OUTPUT_ROOT}/fold{fold_id}/val/images",
        "test": None,
        "nc": nc,
        "names": names
    }

    with open(f"data_plate_fold{i+1}.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

print("\n✔ 5-fold dataset generated using HARDLINK. Very fast. No extra storage needed.\n")
