import os
from collections import Counter

DATASET_ROOT = r"C:\Kuliah\Semester_8\Dataset_YOLO\dataset_plate_1class"

LABEL_FOLDERS = [
    os.path.join(DATASET_ROOT, "train", "labels"),
    os.path.join(DATASET_ROOT, "valid", "labels"),
]

counter = Counter()
total_files = 0
empty_files = 0

for label_dir in LABEL_FOLDERS:
    if not os.path.exists(label_dir):
        print(f"❌ Folder tidak ditemukan: {label_dir}")
        continue

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        total_files += 1
        path = os.path.join(label_dir, file)

        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            cls = int(line.split()[0])
            counter[cls] += 1

print("\n📊 Class distribution:")
for k, v in counter.items():
    print(f"Class {k}: {v}")

print(f"\nTotal label files : {total_files}")
print(f"Empty label files : {empty_files}")
print(f"Total classes found : {len(counter)}")
