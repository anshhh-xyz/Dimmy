import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    yaml_path = str(config.YOLO_DATASET / "data.yaml")
    model = YOLO("yolo11s.pt")  # was " yolo11s.pt" (leading space -> file not found)

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=16,
        workers=2,
        device=config.DEVICE or 0,
        patience=20,
        project=str(config.RUNS_DIR / "detect"),
        # was 'idd_yolov8s_local', but every other script loads idd_yolov11s_local
        name="idd_yolov11s_local",
    )


if __name__ == "__main__":
    main()
