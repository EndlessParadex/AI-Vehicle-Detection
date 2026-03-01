from ultralytics import YOLO
import os, yaml, pandas as pd, shutil

# ===================================================================
# KONFIGURASI EKSPERIMEN
# ===================================================================
EXPERIMENT_NAME = "yolo12s_plate"
MODEL_PATH = "yolo12s.pt"
RUN_DIR = f"runs/{EXPERIMENT_NAME}_5fold"
SUMMARY_DIR = f"summary/{EXPERIMENT_NAME}"
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ===================================================================
# VALIDASI YAML
# ===================================================================
def validate_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML tidak ditemukan: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    required = ["train", "val", "nc", "names"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Field '{key}' tidak ada di YAML: {yaml_path}")

    for split in ["train", "val"]:
        if not os.path.exists(cfg[split]):
            raise RuntimeError(f"Dataset tidak ditemukan: {cfg[split]}")
    return cfg


# ===================================================================
# TRAINING FOLD & SIMPAN METRIK
# ===================================================================
def train_single_fold(fold_idx):
    yaml_path = f"yaml/data_plate_fold{fold_idx}.yaml"
    validate_yaml(yaml_path)

    print(f"\n========== TRAIN PLATE FOLD {fold_idx} ==========")

    model = YOLO(MODEL_PATH)

    # Train dan ambil output result
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        optimizer="AdamW",
        patience=25,
        pretrained=True,
        project=RUN_DIR,
        name=f"fold_{fold_idx}",
    )

    # Ambil metrik langsung dari objek hasil training
    metrics = results.results_dict or {}
    precision  = metrics.get("metrics/precision(B)", 0)
    recall     = metrics.get("metrics/recall(B)", 0)
    map50      = metrics.get("metrics/mAP50(B)", 0)
    map5095    = metrics.get("metrics/mAP50-95(B)", 0)

    # Lokasi best model
    best_model_path = f"{RUN_DIR}/fold_{fold_idx}/weights/best.pt"

    # Simpan ringkasan hasil
    with open(f"{SUMMARY_DIR}/fold_{fold_idx}.txt", "w") as f:
        f.write(
            f"experiment: {EXPERIMENT_NAME}\n"
            f"fold: {fold_idx}\n"
            f"precision: {precision}\n"
            f"recall: {recall}\n"
            f"mAP50: {map50}\n"
            f"mAP50-95: {map5095}\n"
            f"best_model_path: {best_model_path}\n"
        )

    # Salin best model → summary
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, f"{SUMMARY_DIR}/best_fold_{fold_idx}.pt")

    print(f"[SAVED] Fold {fold_idx} → {SUMMARY_DIR}/fold_{fold_idx}.txt")



# ===================================================================
# MAIN
# ===================================================================
def main():
    print(f"\n=== MULAI TRAINING VEHICLE {EXPERIMENT_NAME} ===\n")

    for fold_idx in range(1, 6):
        try:
            train_single_fold(fold_idx)
        except Exception as e:
            print(f"[ERROR] Fold {fold_idx} gagal → {e}")
            break

    print("\n=== TRAINING 5-FOLD VEHICLE SELESAI ===")
    print(f"Hasil tersimpan di folder: {SUMMARY_DIR}\n")


if __name__ == "__main__":
    main()
