# evaluate_resnet18_test.py
import os, csv, argparse
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_results/best.pt")
    parser.add_argument("--data", default="dataset_cnn/valid")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ===== TRANSFORM (HARUS SAMA DENGAN TRAINING) =====
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ===== DATASET =====
    dataset = datasets.ImageFolder(args.data, transform=tf)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0   # ⬅️ FIX UTAMA (AMAN DI WINDOWS)
    )

    print("Class mapping:", dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # ===== LOAD MODEL =====
    num_classes = len(dataset.classes)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # ===== EVALUATION =====
    y_true, y_pred, y_conf = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_conf.extend(confs.cpu().numpy())

    # ===== CLASSIFICATION REPORT =====
    target_names = [idx_to_class[i] for i in range(num_classes)]
    print("\n=== CLASSIFICATION REPORT ===\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    ))

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, num_classes), max(5, num_classes / 1.5)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - ResNet18")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()
    print("Saved confusion_matrix.png")

    # ===== SAVE CSV =====
    with open("predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "pred_label", "confidence"])

        for (path, true_idx), pred_idx, conf in zip(
            dataset.samples, y_pred, y_conf
        ):
            writer.writerow([
                os.path.basename(path),
                idx_to_class[true_idx],
                idx_to_class[pred_idx],
                f"{conf:.4f}"
            ])

    print("Saved predictions.csv")

# ===== ENTRY POINT (WAJIB DI WINDOWS) =====
if __name__ == "__main__":
    main()
