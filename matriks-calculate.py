from ultralytics import YOLO
import os
from multiprocessing import freeze_support

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "runs/plate_1class/yolo12s/weights/best.pt"
DATA_YAML  = "data_plate_1class.yaml"
OUTPUT_TXT = "text_result_yolo/documented_result_yolov12_plate_1class.txt"

def main():
    # =========================================================
    # VALIDASI FILE
    # =========================================================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"YAML tidak ditemukan: {DATA_YAML}")

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    # =========================================================
    # LOAD MODEL
    # =========================================================
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # =========================================================
    # EVALUATION
    # =========================================================
    print("Evaluating model on validation set...")
    results = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=8,
        device=0
    )

    # =========================================================
    # AMBIL METRIK YOLO
    # =========================================================
    metrics = results.results_dict

    precision = float(metrics.get("metrics/precision(B)", 0))
    recall    = float(metrics.get("metrics/recall(B)", 0))
    map50     = float(metrics.get("metrics/mAP50(B)", 0))
    map5095   = float(metrics.get("metrics/mAP50-95(B)", 0))

    # =========================================================
    # HITUNG F1 SCORE (RUMUS RESMI)
    # =========================================================
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    # =========================================================
    # PRINT HASIL
    # =========================================================
    print("\n=== DOCUMENTED MODEL RESULT (YOLOv5 – PLATE) ===")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1_score:.4f}")
    print(f"mAP@50      : {map50:.4f}")
    print(f"mAP@50–95   : {map5095:.4f}")

    # =========================================================
    # SIMPAN KE FILE
    # =========================================================
    with open(OUTPUT_TXT, "w") as f:
        f.write("Documented Model Result – YOLOv5 Plate\n")
        f.write("=======================================\n")
        f.write(f"Model Path  : {MODEL_PATH}\n")
        f.write(f"Dataset     : {DATA_YAML}\n\n")

        f.write(f"Precision   : {precision:.4f}\n")
        f.write(f"Recall      : {recall:.4f}\n")
        f.write(f"F1 Score    : {f1_score:.4f}\n")
        f.write(f"mAP@50      : {map50:.4f}\n")
        f.write(f"mAP@50–95   : {map5095:.4f}\n")

    print(f"\nHasil disimpan ke: {OUTPUT_TXT}")

# =========================================================
# ENTRY POINT (WAJIB WINDOWS)
# =========================================================
if __name__ == "__main__":
    freeze_support()
    main()
