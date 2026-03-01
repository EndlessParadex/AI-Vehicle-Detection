from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="data_vehicle.yaml",

        # =====================
        # CORE
        # =====================
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        optimizer="AdamW",
        patience=25,
        project="runs/vehicle",
        name="yolov8s_vehicle",
    )

if __name__ == "__main__":
    main()
