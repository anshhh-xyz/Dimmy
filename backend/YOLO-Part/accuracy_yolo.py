import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    model_weights = str(config.YOLO_WEIGHTS)
    yaml_path = str(config.YOLO_DATASET / "data.yaml")

    print("[+] Loading trained YOLO model...")
    model = YOLO(model_weights)

    print("\nEvaluating YOLO on Test Set...")
    metrics = model.val(data=yaml_path, split="test")

    print("\n" + "=" * 70)
    print("           YOLO MODEL OVERALL METRICS REPORT")
    print("=" * 70)
    print(f" Precision (P) : {metrics.box.mp * 100:.2f}%")
    print(f" Recall (R)    : {metrics.box.mr * 100:.2f}%")
    print(f" mAP@50        : {metrics.box.map50 * 100:.2f}%")
    print(f" mAP@50-95     : {metrics.box.map * 100:.2f}%")
    print("=" * 70)

    print("\nPER-CLASS DETECTION METRICS TABLE")
    print("-" * 70)
    print(f"{'Class Name':<20} | {'Precision':<12} | {'Recall':<12} | {'mAP@50':<12} | {'mAP@50-95':<12}")
    print("-" * 70)

    names = model.names
    # The per-class arrays only contain classes that actually appear in the
    # test split, in the order given by ap_class_index. The old code indexed
    # them by class id, which printed the wrong numbers next to the wrong
    # names whenever a class was missing from the split.
    per_class = {}
    for row, cls_id in enumerate(metrics.box.ap_class_index):
        per_class[int(cls_id)] = (
            metrics.box.p[row] * 100,
            metrics.box.r[row] * 100,
            metrics.box.ap50[row] * 100,
            metrics.box.ap[row] * 100,
        )

    for cls_id in sorted(names):
        p, r, map50, map_all = per_class.get(cls_id, (0.0, 0.0, 0.0, 0.0))
        note = "" if cls_id in per_class else "  (no instances in split)"
        print(f"{names[cls_id]:<20} | {p:>10.2f}% | {r:>10.2f}% | {map50:>10.2f}% | {map_all:>10.2f}%{note}")

    print("=" * 70)


if __name__ == "__main__":
    main()
