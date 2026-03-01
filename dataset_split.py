import os
import random
import shutil

# =====================
# CONFIG
# =====================
SRC_ROOT = r"C:\Kuliah\Semester_8\Dataset_YOLO\dataset_plate_640_70-30"
DST_ROOT = r"C:\Kuliah\Semester_8\Dataset_YOLO\dataset_plate_1class_resplit"

TRAIN_RATIO = 0.8
SEED = 42
random.seed(SEED)

# =====================
# COLLECT ALL DATA
# =====================
all_pairs = []

for split in ["train", "valid"]:
    img_dir = os.path.join(SRC_ROOT, split, "images")
    lbl_dir = os.path.join(SRC_ROOT, split, "labels")

    for img in os.listdir(img_dir):
        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        lbl = os.path.splitext(img)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl)

        all_pairs.append((
            os.path.join(img_dir, img),
            lbl_path if os.path.exists(lbl_path) else None
        ))

print(f"Total images collected: {len(all_pairs)}")

# =====================
# SHUFFLE & SPLIT
# =====================
random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * TRAIN_RATIO)

train_pairs = all_pairs[:split_idx]
val_pairs   = all_pairs[split_idx:]

# =====================
# CREATE OUTPUT DIR
# =====================
for p in [
    "train/images", "train/labels",
    "valid/images", "valid/labels"
]:
    os.makedirs(os.path.join(DST_ROOT, p), exist_ok=True)

def copy_pairs(pairs, split):
    for img_path, lbl_path in pairs:
        shutil.copy2(
            img_path,
            os.path.join(DST_ROOT, split, "images", os.path.basename(img_path))
        )

        lbl_dst = os.path.join(
            DST_ROOT, split, "labels",
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        if lbl_path:
            shutil.copy2(lbl_path, lbl_dst)
        else:
            open(lbl_dst, "w").close()  # label kosong tetap valid

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "valid")

# =====================
# REPORT
# =====================
print("✅ Dataset re-split selesai")
print(f"Train images : {len(train_pairs)}")
print(f"Valid images : {len(val_pairs)}")
