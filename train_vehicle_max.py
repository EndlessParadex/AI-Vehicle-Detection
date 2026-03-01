from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="data_vehicle.yaml",

        # ===== CORE =====
        epochs=180,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,

        # ===== OPTIMIZER =====
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=5e-4,

        # ===== REGULARIZATION =====
        label_smoothing=0.05,
        dropout=0.0,

        # ===== AUGMENTATION (SMALL OBJECT FRIENDLY) =====
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,

        # ===== MULTI SCALE =====
        multi_scale=True,

        # ===== TRAIN CONTROL =====
        patience=30,
        cos_lr=True,

        name="yolov8s_vehicle_max"
    )

if __name__ == "__main__":
    main()
