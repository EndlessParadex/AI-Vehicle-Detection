import os
from collections import Counter

# ===============================
# KONFIGURASI
# ===============================
LABEL_DIR = r"full-training-for-yolo_plate/valid/labels"
NUM_CLASSES = 7

CLASS_NAMES = [
    "plate-ev-black",
    "plate-ev-white",
    "plate-ev-yellow",
    "plate-goverment",
    "plate-ice-black",
    "plate-ice-white",
    "plate-public"
]

# ===============================
# PROSES HITUNG
# ===============================
class_counter = Counter()
total_objects = 0
label_files = 0

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    label_files += 1
    file_path = os.path.join(LABEL_DIR, file)

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            class_id = int(line.split()[0])
            class_counter[class_id] += 1
            total_objects += 1

# ===============================
# OUTPUT
# ===============================
print("\n📊 VALIDATION SET STATISTICS")
print("=" * 40)
print(f"Total label files   : {label_files}")
print(f"Total objects (bbox): {total_objects}")
print("-" * 40)

for class_id in range(NUM_CLASSES):
    count = class_counter.get(class_id, 0)
    name = CLASS_NAMES[class_id]
    print(f"Class {class_id} ({name:<18}) : {count} objects")

print("=" * 40)
