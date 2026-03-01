from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt") 

    model.train(
        data="data_plate.yaml",
        epochs=150,
        imgsz=640,               # lebih besar karena menangani small object
        batch=8,                 # GPU RTX 4050 4GB aman
        device=0,
        workers=4,
        optimizer="AdamW",
        patience=0,
    )

if __name__ == "__main__":
    main()
