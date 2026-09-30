import os
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    model = YOLO(str(config.YOLO_WEIGHTS))

    input_image = sys.argv[1] if len(sys.argv) > 1 else None
    if input_image is None:
        test_dir = config.YOLO_DATASET / "test" / "images"
        imgs = sorted(test_dir.glob("*.jpg")) if test_dir.exists() else []
        input_image = str(imgs[0]) if imgs else None

    if not input_image or not os.path.exists(input_image):
        print(f"Image not found: {input_image}. Usage: python test_predict_yolo.py <image>")
        return

    print(f"Running detection on: {os.path.basename(input_image)}")
    results = model(input_image, conf=0.25)

    print("\n--- Detection Results ---")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"- {label:15s} | Confidence: {conf:.2f} | BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    output_filename = "prediction_output.jpg"
    results[0].save(filename=output_filename)
    print(f"\nAnnotated result saved as '{output_filename}'")

    try:
        results[0].show()
    except Exception as exc:  # headless machines have no display
        print(f"(could not open a preview window: {exc})")


if __name__ == "__main__":
    main()
