import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "dataset_cnn"
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "cnn_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# TRANSFORMS
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "valid"), transform=val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Class mapping:", train_ds.class_to_idx)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# =========================
# TRAINING
# =========================
best_acc = 0.0
log_file = open(os.path.join(SAVE_DIR, "metrics.txt"), "w")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0)

    avg_loss = running_loss / len(train_loader)

    msg = (
        f"Epoch {epoch+1} | "
        f"Loss {avg_loss:.4f} | "
        f"Acc {acc:.4f} | "
        f"Prec {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )

    print(msg)
    log_file.write(msg + "\n")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
        print("✅ Best model saved")

log_file.close()
print(f"Training selesai. Best Val Acc: {best_acc:.4f}")
