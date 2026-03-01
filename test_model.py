from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r"runs/plate_1class/yolov8s_6404/weights/best.pt")

# Input video
video_path = "video/VIDEO_1.MOV"
cap = cv2.VideoCapture(video_path)

# Ambil FPS dan ukuran frame original
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Siapkan penulis video output (MP4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_yolo14.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prediksi
    results = model(frame)

    # Annotasi bounding box
    annotated = results[0].plot()

    # Simpan frame ke output video
    out.write(annotated)

    # Tampilkan ke layar
    cv2.imshow("YOLO Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
