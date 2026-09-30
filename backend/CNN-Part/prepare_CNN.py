import os
import sys
from pathlib import Path
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def main():
    base_dir = str(config.CNN_RAW_DATASET)
    output_dir = str(config.CNN_DATASET)
    
    yaml_path = os.path.join(base_dir, "data.yaml")
    print(f"Reading class names from {yaml_path}...")
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    class_names = data_cfg["names"]
    if isinstance(class_names, dict):  # data.yaml may store names as {0: name, ...}
        class_names = [class_names[k] for k in sorted(class_names)]
    print(f"Found {len(class_names)} traffic sign classes.")

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        
        if not os.path.exists(img_dir):
            continue

        out_split = "val" if split == "valid" else split
        print(f"\n Cropping & segregating signs for '{out_split}' split...")

        total_crops = 0
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                crop = img[y1:y2, x1:x2]
                if cls_id >= len(class_names):
                    continue
                class_name = class_names[cls_id]

                class_dir = os.path.join(output_dir, out_split, class_name)
                os.makedirs(class_dir, exist_ok=True)

                crop_name = f"{os.path.splitext(img_name)[0]}_{idx}.jpg"
                cv2.imwrite(os.path.join(class_dir, crop_name), crop)
                total_crops += 1

        print(f"  → Created {total_crops} cropped sign images for '{out_split}'.")

    print(f"\n All done! Segregated dataset ready at: {output_dir}")

if __name__ == "__main__":
    main()
