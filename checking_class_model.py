import torch

ckpt = torch.load("faster_rcnn_vehicle_mobilenetv3_320/best.pth", map_location="cpu")

for k, v in ckpt.items():
    if "cls_score.weight" in k:
        print("cls_score.weight shape:", v.shape)
