import os
from ultralytics import YOLO

def main():
    model_weights = r"C:\ME\Project-Detection\runs\detect\idd_yolov11s_local\weights\best.pt"
    yaml_path = r"C:\ME\Project-Detection\IDDDetectionsYOLODataset_subset\data.yaml"

    print("[+] Loading trained YOLO model...")
    model = YOLO(model_weights)

    print("\n📊 Evaluating YOLO on Test Set...")
    metrics = model.val(data=yaml_path, split='test')

    print("\n" + "=" * 70)
    print("           🎯 YOLO MODEL OVERALL METRICS REPORT")
    print("=" * 70)
    print(f" Precision (P) : {metrics.box.mp * 100:.2f}%")
    print(f" Recall (R)    : {metrics.box.mr * 100:.2f}%")
    print(f" mAP@50        : {metrics.box.map50 * 100:.2f}%")
    print(f" mAP@50-95     : {metrics.box.map * 100:.2f}%")
    print("=" * 70)

    print("\n📋 PER-CLASS DETECTION METRICS TABLE")
    print("-" * 70)
    print(f"{'Class Name':<20} | {'Precision':<12} | {'Recall':<12} | {'mAP@50':<12} | {'mAP@50-95':<12}")
    print("-" * 70)

    names = model.names
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap

    for i in range(len(names)):
        cls_name = names[i]
        p = p_per_class[i] * 100 if i < len(p_per_class) else 0.0
        r = r_per_class[i] * 100 if i < len(r_per_class) else 0.0
        map50 = ap50_per_class[i] * 100 if i < len(ap50_per_class) else 0.0
        map_all = ap_per_class[i] * 100 if i < len(ap_per_class) else 0.0

        print(f"{cls_name:<20} | {p:>10.2f}% | {r:>10.2f}% | {map50:>10.2f}% | {map_all:>10.2f}%")

    print("=" * 70)

if __name__ == '__main__':
    main()