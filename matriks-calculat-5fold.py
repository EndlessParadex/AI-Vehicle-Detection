import numpy as np
from ultralytics import YOLO

def main():
    map50_list = []
    map_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(1, 6):
        print(f"\n=== Evaluating Fold {i} ===")

        model = YOLO(f"runs/vehicle_5fold/fold_{i}/weights/best.pt")

        metrics = model.val(
            data=f"data_fold{i}.yaml",
            imgsz=640,
            device=0,
            workers=0,     # ← PENTING untuk Windows
            verbose=False
        )

        mp = metrics.box.mp
        mr = metrics.box.mr
        map50 = metrics.box.map50
        map95 = metrics.box.map

        f1 = 2 * mp * mr / (mp + mr + 1e-16)

        map50_list.append(map50)
        map_list.append(map95)
        precision_list.append(mp)
        recall_list.append(mr)
        f1_list.append(f1)

        print(f"mAP50     : {map50:.4f}")
        print(f"mAP50-95  : {map95:.4f}")
        print(f"Precision : {mp:.4f}")
        print(f"Recall    : {mr:.4f}")
        print(f"F1-score  : {f1:.4f}")

    print("\n===== FINAL 5-FOLD RESULT =====")
    print(f"Mean mAP50     : {np.mean(map50_list):.4f}")
    print(f"Mean mAP50-95  : {np.mean(map_list):.4f}")
    print(f"Mean Precision : {np.mean(precision_list):.4f}")
    print(f"Mean Recall    : {np.mean(recall_list):.4f}")
    print(f"Mean F1-score  : {np.mean(f1_list):.4f}")

if __name__ == "__main__":
    main()
