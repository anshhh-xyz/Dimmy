import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

MAIN_DATASET = str(config.YOLO_SOURCE_DATASET)
FINAL_DATASET = str(config.YOLO_DATASET)
TRAINING_IMAGES = 10000

if not os.path.exists(MAIN_DATASET):
    # Previously this only printed a message and then crashed a few lines later.
    sys.exit(f"DATASET NOT FOUND: {MAIN_DATASET}\n"
             f"Put the IDD YOLO dataset there or set DIMMY_YOLO_SOURCE_DATASET.")

for item in ["train", "test", "val"]:
    os.makedirs(os.path.join(FINAL_DATASET, item, "images"), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DATASET, item, "labels"), exist_ok=True)


def copy_pair(split, img):
    shutil.copy(os.path.join(MAIN_DATASET, split, "images", img),
                os.path.join(FINAL_DATASET, split, "images", img))
    label = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(MAIN_DATASET, split, "labels", label)
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(FINAL_DATASET, split, "labels", label))


for item in ["val", "test"]:
    images = os.listdir(os.path.join(MAIN_DATASET, item, "images"))
    print(f"Copying {len(images)} {item} images...")
    for img in images:
        copy_pair(item, img)

random.seed(42)
train_imgs = os.listdir(os.path.join(MAIN_DATASET, "train", "images"))
subset = random.sample(train_imgs, min(TRAINING_IMAGES, len(train_imgs)))

print(f"Copying {len(subset)} train images")
for img in subset:
    copy_pair("train", img)

# 'path:' must be lowercase - Ultralytics ignores 'Path:' and then resolves the
# relative train/val/test folders against the wrong directory.
yaml_content = f"""path: {Path(FINAL_DATASET).as_posix()}
names:
  0: animal
  1: autorickshaw
  2: bicycle
  3: bus
  4: car
  5: caravan
  6: motorcycle
  7: person
  8: rider
  9: traffic light
  10: traffic sign
  11: trailer
  12: train
  13: truck
  14: vehicle fallback
nc: 15
test: test/images
train: train/images
val: val/images
"""

yaml_path = os.path.join(FINAL_DATASET, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print(f"Wrote {yaml_path}")
