import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#   DATASET COCO ROBOTFLOW (CUSTOM)
# ============================================================
class RoboflowRetinaDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

        # Mapping kategori COCO yang id-nya dimulai dari 0 → harus dimulai dari 1
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_remap = {cid: (i + 1) for i, cid in enumerate(self.cat_ids)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

        # Load annotation
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []

        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])

            # Remap category
            labels.append(self.cat_remap[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
#   TRAINING LOOP
# ============================================================
def train():
    train_dataset = RoboflowRetinaDataset(
        img_folder="dataset_plate_coco/train/images",
        ann_file="dataset_plate_coco/train/_annotations.coco.json",
        transforms=ToTensor()
    )

    valid_dataset = RoboflowRetinaDataset(
        img_folder="dataset_plate_coco/valid/images",
        ann_file="dataset_plate_coco/valid/_annotations.coco.json",
        transforms=ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Jumlah kelas = kategori unik (tanpa background)
    num_classes = len(train_dataset.cat_ids) + 1

    # =======================================================
    #   Load RetinaNet50 FPN Pretrained COCO
    # =======================================================
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None,                       # disable COCO pretrained head
        weights_backbone="DEFAULT",         # use pretrained backbone only
        num_classes=num_classes
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # =======================================================
    #   Training Loop
    # =======================================================
    for epoch in range(30):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "retinanet_resnet50.pth")
    print("Model saved: retinanet_resnet50.pth")


if __name__ == "__main__":
    train()
