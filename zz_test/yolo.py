from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).parent.resolve()

    data_yaml = root / "data.yaml"
    project_dir = root
    run_name = "runs/bottle/cls"

    model = YOLO("yolo11n.pt").to("cuda")
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(project_dir),
        name=run_name,
        pretrained=True,
        # patience = 10,
        # verbose=True
    )
