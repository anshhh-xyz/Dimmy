import os
from ultralytics import YOLO

def main():
    model_weights = r"C:\ME\Project-Detection\runs\detect\idd_yolov11s_local\weights\best.pt"
    model = YOLO(model_weights)

    input_image = r"C:\ME\Project-Detection\IDDDetectionsYOLODataset_subset\test\images\HYD-2018-08-24_13-22-50_0008384.jpg"

    if not os.path.exists(input_image):
        print(f"❌ Image not found at: {input_image}")
        return

    print(f"🔍 Running detection on: {os.path.basename(input_image)}")
    results = model(input_image, conf=0.25)

    print("\n--- Detection Results ---")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"• {label:15s} | Confidence: {conf:.2f} | BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    output_filename = "prediction_output.jpg"
    results[0].save(filename=output_filename)
    print(f"\n✅ Annotated result saved as '{output_filename}'")

    results[0].show()

if __name__ == '__main__':
    main()
