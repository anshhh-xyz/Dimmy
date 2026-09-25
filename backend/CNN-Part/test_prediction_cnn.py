import os
import sys
import json
import random
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_class_names(output_dir, checkpoint):
    if "class_names" in checkpoint:
        return checkpoint["class_names"]
    
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            if "idx_to_class" in mapping:
                idx_map = mapping["idx_to_class"]
                return [idx_map[str(i)] for i in range(len(idx_map))]
            elif "class_to_idx" in mapping:
                cls_map = mapping["class_to_idx"]
                sorted_classes = sorted(cls_map.items(), key=lambda x: x[1])
                return [cls[0] for cls in sorted_classes]
    
    raise ValueError("Could not find class names in model checkpoint or class_mapping.json")

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    
    print(f"[+] Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    output_dir = os.path.dirname(checkpoint_path)
    class_names = load_class_names(output_dir, checkpoint)
    num_classes = len(class_names)
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    val_acc = checkpoint.get("val_acc", None)
    epoch = checkpoint.get("epoch", None)
    if val_acc is not None:
        print(f"[*] Loaded model checkpoint from Epoch {epoch} | Saved Val Acc: {val_acc * 100:.2f}%")
        
    return model, class_names

def get_random_test_image(dataset_dir):
    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_dir):
        test_dir = dataset_dir
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, f))
                
    if not image_paths:
        raise FileNotFoundError(f"No test images found in dataset path: {test_dir}")
        
    return random.choice(image_paths)

def predict_single_image(model, class_names, img_path, device, img_size=128, top_k=5):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    top_k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class_idx": idx.item(),
            "class_name": class_names[idx.item()],
            "confidence": prob.item() * 100
        })
        
    return results

def annotate_image(img_path, predictions, ground_truth=None):
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        return None
        
    h, w, _ = cv_img.shape
    
    display_h = max(h * 4, 300)
    display_w = max(w * 4, 400)
    resized_img = cv2.resize(cv_img, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
    
    banner_height = 110
    canvas = np.zeros((display_h + banner_height, display_w, 3), dtype=np.uint8) + 30
    canvas[banner_height:, :] = resized_img
    
    top_pred = predictions[0]
    pred_label = top_pred["class_name"]
    confidence = top_pred["confidence"]
    
    is_correct = (ground_truth is not None) and (ground_truth.lower() == pred_label.lower())
    title_text = f"Pred: {pred_label} ({confidence:.1f}%)"
    
    if ground_truth:
        gt_text = f"Ground Truth: {ground_truth}"
        color = (0, 230, 0) if is_correct else (0, 70, 255)
    else:
        gt_text = "Ground Truth: Unknown"
        color = (255, 200, 0)
        
    cv2.putText(canvas, title_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, gt_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"File: {os.path.basename(img_path)}", (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Predict single traffic sign image using EfficientNet CNN")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (default: random test image)")
    parser.add_argument("--model", type=str, default=r"C:\ME\Project-Detection\runs\efficientnet_train\best_efficientnet.pth", help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\ME\Project-Detection\CNN_Classification_Dataset", help="Path to dataset root")
    parser.add_argument("--top_k", type=int, default=5, help="Top K predictions to show")
    parser.add_argument("--show", action="store_true", help="Display visual prediction image window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    model, class_names = load_model(args.model, device)

    if args.image and os.path.exists(args.image):
        img_path = args.image
        parent_dir = os.path.basename(os.path.dirname(img_path))
        ground_truth = parent_dir if parent_dir in class_names else None
    else:
        if args.image:
            print(f"[!] Warning: Specified image path '{args.image}' not found. Selecting random test image...")
        img_path = get_random_test_image(args.dataset_dir)
        parent_dir = os.path.basename(os.path.dirname(img_path))
        ground_truth = parent_dir if parent_dir in class_names else None

    print(f"\n[+] Image selected: {img_path}")
    if ground_truth:
        print(f"[*] Ground Truth Label: {ground_truth}")

    predictions = predict_single_image(model, class_names, img_path, device, img_size=128, top_k=args.top_k)

    print("\n" + "=" * 65)
    print(" CNN PREDICTION RESULTS")
    print("=" * 65)
    print(f" Top Prediction : {predictions[0]['class_name']} ({predictions[0]['confidence']:.2f}%)")
    if ground_truth:
        status = "MATCH / CORRECT [OK]" if ground_truth.lower() == predictions[0]['class_name'].lower() else "MISMATCH / INCORRECT [X]"
        print(f" Result Status  : {status}")
    print("-" * 65)
    print(f"{'Rank':<6} | {'Class Name':<42} | {'Confidence':<10}")
    print("-" * 65)
    for i, pred in enumerate(predictions, 1):
        print(f"#{i:<5} | {pred['class_name']:<42} | {pred['confidence']:>8.2f}%")
    print("=" * 65)

    annotated = annotate_image(img_path, predictions, ground_truth)
    if annotated is not None:
        output_dir = os.path.dirname(args.model)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "prediction_sample.png")
        cv2.imwrite(save_path, annotated)
        print(f"\n[+] Visual prediction output saved to: {save_path}")
        
        if args.show:
            cv2.imshow("CNN Traffic Sign Prediction", annotated)
            print("Press any key on the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
