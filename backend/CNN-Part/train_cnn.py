import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    dataset_dir = r"C:\ME\Project-Detection\CNN_Classification_Dataset"
    output_dir = r"C:\ME\Project-Detection\runs\efficientnet_train"
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = 30
    batch_size = 32
    learning_rate = 0.001
    img_size = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"📂 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Classes: {num_classes}")

    class_mapping = {"class_to_idx": train_dataset.class_to_idx, "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()}}
    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("🏗️ Loading Pretrained EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_efficientnet.pth")

    print(f"\n🚀 Starting training for {epochs} epochs...\n" + "=" * 50)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_dataset)
        train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        val_loss = val_loss / len(val_dataset)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc
            }, best_model_path)
            print(f"  ⭐ Best Model Saved! (Val Acc: {val_acc*100:.2f}%)")

    print(f"\n🎉 Training complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"💾 Checkpoint saved to: {best_model_path}")

if __name__ == "__main__":
    main()
