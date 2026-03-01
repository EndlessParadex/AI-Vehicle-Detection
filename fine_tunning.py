from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/Yolov8_Plate/weights/best.pt")  # bukan yolov8s.pt lagi

    model.train(
        data="data_plate.yaml",
        epochs=60,              
        imgsz=1280,
        batch=8,
        device=0,
        workers=4,
        half=True,
        
        freeze=10,              # Pretrained Knowledge Tidak Hilang

        # LOWER learning rate (VERY important for fine tuning)
        lr0=0.0005,
        lrf=0.001,

        # Augment Ringan
        mosaic=0.4,
        copy_paste=0.1,
        flipud=0.1,
        degrees=5,
        scale=0.7,
        
        patience=15,
    )

if __name__ == "__main__":
    main()
