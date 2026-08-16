from ultralytics import YOLO


def main():
    
    yaml_path = r"C:\ME\Project-Detection\IDDDetectionsYOLODataset_subset\data.yaml"
    model=YOLO(" yolo11s.pt")

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=16,
        workers=2,
        device=0,
        patience=20,
        project='runs/detect',
        name='idd_yolov8s_local'
        )

if __name__ == '__main__':
    main()