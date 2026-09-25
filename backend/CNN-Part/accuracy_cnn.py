import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class MasterClassDataset(Dataset):
    def __init__(self, root, master_class_to_idx, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = list(master_class_to_idx.keys())
        self.class_to_idx = master_class_to_idx

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for folder in os.listdir(root):
            fpath = os.path.join(root, folder)
            if not os.path.isdir(fpath) or folder not in master_class_to_idx:
                continue
            target_idx = master_class_to_idx[folder]
            for fname in os.listdir(fpath):
                if fname.lower().endswith(valid_extensions):
                    self.samples.append((os.path.join(fpath, fname), target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

def load_master_mapping(output_dir, checkpoint):
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            if "class_to_idx" in mapping:
                cls_to_idx = mapping["class_to_idx"]
                sorted_classes = sorted(cls_to_idx.items(), key=lambda x: x[1])
                class_names = [cls[0] for cls in sorted_classes]
                return cls_to_idx, class_names
    
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        cls_to_idx = {name: idx for idx, name in enumerate(class_names)}
        return cls_to_idx, class_names

    raise ValueError("Could not find class mapping in model checkpoint or class_mapping.json")

def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    
    print(f"[+] Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    output_dir = os.path.dirname(checkpoint_path)
    class_to_idx, class_names = load_master_mapping(output_dir, checkpoint)
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
    
    return model, class_to_idx, class_names, checkpoint

def evaluate_model(model, data_loader, num_classes, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_sz = labels.size(0)
            total_loss += loss.item() * batch_sz
            total_samples += batch_sz

            _, preds = torch.max(outputs, 1)
            top1_correct += (preds == labels).sum().item()

            maxk = min(5, num_classes)
            _, top5_preds = outputs.topk(maxk, 1, True, True)
            top5_correct += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    top1_acc = (top1_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    top5_acc = (top5_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    return {
        "loss": avg_loss,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "total_samples": total_samples,
        "confusion_matrix": confusion_matrix
    }

def compute_per_class_metrics(confusion_matrix, class_names):
    num_classes = len(class_names)
    metrics = []

    total_tp = 0
    total_support = 0

    for i in range(num_classes):
        tp = confusion_matrix[i, i].item()
        fp = (confusion_matrix[:, i].sum() - tp).item()
        fn = (confusion_matrix[i, :].sum() - tp).item()
        support = confusion_matrix[i, :].sum().item()

        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append({
            "class_idx": i,
            "class_name": class_names[i],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "correct": tp
        })

        total_tp += tp
        total_support += support

    valid_classes = [m for m in metrics if m["support"] > 0]
    macro_prec = sum(m["precision"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0
    macro_rec = sum(m["recall"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0
    macro_f1 = sum(m["f1_score"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0

    weighted_prec = sum(m["precision"] * m["support"] for m in metrics) / total_support if total_support > 0 else 0.0
    weighted_rec = sum(m["recall"] * m["support"] for m in metrics) / total_support if total_support > 0 else 0.0
    weighted_f1 = sum(m["f1_score"] * m["support"] for m in metrics) / total_support if total_support > 0 else 0.0

    summary = {
        "macro_avg": {"precision": macro_prec, "recall": macro_rec, "f1_score": macro_f1},
        "weighted_avg": {"precision": weighted_prec, "recall": weighted_rec, "f1_score": weighted_f1}
    }

    return metrics, summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Accuracy Scores for Best Trained EfficientNet CNN Model")
    parser.add_argument("--model", type=str, default=r"C:\ME\Project-Detection\runs\efficientnet_train\best_efficientnet.pth", help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\ME\Project-Detection\CNN_Classification_Dataset", help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "all"], help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=128, help="Image resize dimension")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    model, class_to_idx, class_names, checkpoint = load_model(args.model, device)
    num_classes = len(class_names)

    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_dir = os.path.join(args.dataset_dir, args.split)
    if not os.path.exists(eval_dir):
        print(f"[!] Split path '{eval_dir}' not found. Checking root dataset path...")
        eval_dir = args.dataset_dir

    print(f"[+] Loading dataset split '{args.split}' from: {eval_dir}")
    dataset = MasterClassDataset(eval_dir, master_class_to_idx=class_to_idx, transform=val_transforms)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"[*] Total Evaluation Samples: {len(dataset)} | Total Model Classes: {num_classes}")

    print(f"\n[+] Running model evaluation on {device}...")
    eval_res = evaluate_model(model, data_loader, num_classes, device)
    class_metrics, summary_metrics = compute_per_class_metrics(eval_res["confusion_matrix"], class_names)

    print("\n" + "=" * 70)
    print(" BEST EFFICIENTNET MODEL ACCURACY EVALUATION SUMMARY")
    print("=" * 70)
    if "epoch" in checkpoint and "val_acc" in checkpoint:
        print(f" Checkpoint Saved Epoch    : {checkpoint['epoch']}")
        print(f" Checkpoint Saved Val Acc  : {checkpoint['val_acc'] * 100:.2f}%")
    print(f" Target Split Evaluated    : {args.split.upper()}")
    print(f" Total Test Samples       : {eval_res['total_samples']}")
    print(f" Test Loss (CrossEntropy) : {eval_res['loss']:.4f}")
    print(f" Top-1 Accuracy            : {eval_res['top1_acc']:.2f}%")
    print(f" Top-5 Accuracy            : {eval_res['top5_acc']:.2f}%")
    print("-" * 70)
    print(f" Macro Average Precision   : {summary_metrics['macro_avg']['precision']:.2f}%")
    print(f" Macro Average Recall      : {summary_metrics['macro_avg']['recall']:.2f}%")
    print(f" Macro Average F1-Score    : {summary_metrics['macro_avg']['f1_score']:.2f}%")
    print("-" * 70)
    print(f" Weighted Average F1-Score : {summary_metrics['weighted_avg']['f1_score']:.2f}%")
    print("=" * 70)

    print("\n PER-CLASS CLASSIFICATION METRICS REPORT")
    print("-" * 75)
    print(f"{'Class Name':<38} | {'Support':<8} | {'Correct':<8} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for m in class_metrics:
        if m['support'] > 0:
            acc = (m['correct'] / m['support'] * 100.0)
            print(f"{m['class_name']:<38} | {m['support']:<8} | {m['correct']:<8} | {acc:>8.2f}% | {m['f1_score']:>8.2f}%")
    print("=" * 75)

    report_data = {
        "model_checkpoint": args.model,
        "split": args.split,
        "total_samples": eval_res["total_samples"],
        "top1_accuracy": eval_res["top1_acc"],
        "top5_accuracy": eval_res["top5_acc"],
        "loss": eval_res["loss"],
        "macro_avg": summary_metrics["macro_avg"],
        "weighted_avg": summary_metrics["weighted_avg"],
        "per_class_metrics": class_metrics
    }

    output_dir = os.path.dirname(args.model)
    report_file = os.path.join(output_dir, f"accuracy_report_{args.split}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"\n[+] Detailed accuracy report saved to: {report_file}")

if __name__ == "__main__":
    main()
