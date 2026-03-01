import os
import shutil
import random
from glob import glob

# === KONFIGURASI ===
ROOT = "dataset_vehicle_640_70-30"
OUT = "dataset_5fold"
random.seed(42)

# Gabungkan semua data (train dan valid) menjadi satu list
all_images = glob(os.path.join(ROOT, "train/images/*.jpg")) + \
             glob(os.path.join(ROOT, "valid/images/*.jpg"))

print("Total images:", len(all_images))  # harusnya 5700

# Shuffle dataset
random.shuffle(all_images)

# Bagi menjadi 5 folds
fold_size = len(all_images) // 5
folds = [all_images[i*fold_size:(i+1)*fold_size] for i in range(5)]

# Jika sisa (tidak habis dibagi 5), masukkan ke fold terakhir
remaining = len(all_images) - fold_size*5
if remaining > 0:
    folds[-1] += all_images[-remaining:]

# ---- Generate 5 fold directories ----
for i in range(5):
    fold_dir = os.path.join(OUT, f"fold{i+1}")
    train_img_dir = os.path.join(fold_dir, "train/images")
    train_lbl_dir = os.path.join(fold_dir, "train/labels")
    val_img_dir = os.path.join(fold_dir, "val/images")
    val_lbl_dir = os.path.join(fold_dir, "val/labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Val set = fold ke-i
    val_set = folds[i]

    # Train set = semua fold kecuali fold ke-i
    train_set = [img for j, f in enumerate(folds) if j != i for img in f]

    # Copy data
    for img_path in train_set:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        shutil.copy(img_path, train_img_dir)
        shutil.copy(label_path, train_lbl_dir)

    for img_path in val_set:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        shutil.copy(img_path, val_img_dir)
        shutil.copy(label_path, val_lbl_dir)

    print(f"Fold {i+1}: train={len(train_set)}, val={len(val_set)}")

print("Selesai: dataset 5-fold dibuat di:", OUT)
