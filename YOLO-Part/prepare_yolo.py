import os,shutil,random

MAIN_DATASET=r"C:\ME\Project-Detection\IDDDetectionsYOLODataset"
MAIN_PATH=r"C:\ME\Project-Detection"
FINAL_DATASET="random_string_for_now"

if os.path.exists (MAIN_DATASET):
    os.makedirs(os.path.join(MAIN_PATH,"IDDDetectionsYOLODataset_subset"), exist_ok=True)
    FINAL_DATASET=os.path.join(MAIN_PATH,"IDDDetectionsYOLODataset_subset")
    for item in ['train' , 'test' , 'val']:
        os.makedirs(os.path.join(FINAL_DATASET , item , "images" ), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DATASET , item , "labels" ), exist_ok=True)
else:
    print("DATASET NOT FOUND")


for item in ['val' , 'test']:
    images=os.listdir(os.path.join(MAIN_DATASET , item , 'images' ))
    print(f"Copying {len(images)} {item} images...")
    for img in images:
        shutil.copy( os.path.join(MAIN_DATASET , item , 'images',img) , os.path.join(FINAL_DATASET , item , 'images',img))
        label = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(MAIN_DATASET, item, 'labels', label)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(FINAL_DATASET, item, 'labels', label))


random.seed(42)
TRAINING_IMAGES=10000
train_imgs=os.listdir(os.path.join(MAIN_DATASET,'train','images'))
subset=random.sample(train_imgs, min(10000, len(train_imgs)))


print("Copying 10k images")
for img in subset:
    shutil.copy(os.path.join(MAIN_DATASET , 'train' , 'images',img) , os.path.join(FINAL_DATASET , 'train' , 'images',img))
    label=os.path.splitext(img)[0]+'.txt'
    label_path=os.path.join(MAIN_DATASET, 'train','labels',label)
    if os.path.exists(label_path):
        shutil.copy(label_path,os.path.join(FINAL_DATASET,'train','labels',label))

yaml_content =f"""Path: {FINAL_DATASET}
names:
- animal
- autorickshaw
- bicycle
- bus
- car
- caravan
- motorcycle
- person
- rider
- traffic light
- traffic sign
- trailer
- train
- truck
- vehicle fallback
nc: 15
test: test/images
train: train/images
val: val/images
"""

yaml_path = os.path.join(FINAL_DATASET, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)