from ultralytics import YOLO
import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26s.pt")

    results = model.train(
        data="/home/martina/maritime/yolo_model/v2_yolo26/data.yaml",
        imgsz=640,
        epochs=150,
        batch=24,
        device=0,
        patience=0,
        project="/home/martina/maritime/yolo_model/runs/",
        name="mines_training_yolo26"
    )

    print("Training complete.")
    print("Best model:", model.trainer.best)

if __name__ == "__main__":
    main()
