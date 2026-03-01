from ultralytics import YOLO

def main():
    model = YOLO("yolo12s.pt")

    model.train(
        data="data_plate_1class.yaml",
        epochs=100,
        imgsz=640,
        batch=8,              
        device=0,
        optimizer="AdamW",
        patience=25,
        workers=32,
        project="runs/plate_1class",
        name="yolo12s"
    )

if __name__ == "__main__":
    main()
