from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "tampered"
MASK_DIR = PROJECT_ROOT / "data" / "masks"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_unet.pth"

IMAGE_SIZE = 256
THRESHOLD = 0.5


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


def calculate_metrics(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    pred_pixels = pred.sum()
    target_pixels = target.sum()

    if union == 0:
        iou = 1.0 if pred_pixels == 0 else 0.0
    else:
        iou = intersection / union

    denominator = pred_pixels + target_pixels

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2 * intersection) / denominator

    if pred_pixels == 0:
        precision = 1.0 if target_pixels == 0 else 0.0
    else:
        precision = intersection / pred_pixels

    if target_pixels == 0:
        recall = 1.0 if pred_pixels == 0 else 0.0
    else:
        recall = intersection / target_pixels

    return iou, dice, precision, recall


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading U-Net...")

    model = UNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Checkpoint loaded successfully.")
    print()

    image_files = sorted(
        IMAGE_DIR.glob("*_tampered.*")
    )

    print("Tampered images found:", len(image_files))

    ious = []
    dices = []
    precisions = []
    recalls = []

    skipped = 0

    with torch.no_grad():

        for image_path in tqdm(
            image_files,
            desc="Evaluating localization"
        ):

            base_name = image_path.stem.replace("_tampered", "")
            mask_path = MASK_DIR / f"{base_name}_mask.png"

            if not mask_path.exists():
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")

            image_resized = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image_np = np.array(
                image_resized
            ).astype(np.float32) / 255.0

            image_np = (
                image_np - np.array(
                    [0.485, 0.456, 0.406],
                    dtype=np.float32
                )
            ) / np.array(
                [0.229, 0.224, 0.225],
                dtype=np.float32
            )

            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1)
            ).unsqueeze(0).float().to(device)

            output = model(image_tensor)

            prediction = torch.sigmoid(output)

            prediction = (
                prediction.squeeze()
                .cpu()
                .numpy()
            )

            prediction_binary = (
                prediction >= THRESHOLD
            ).astype(np.uint8)

            target = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if target is None:
                skipped += 1
                continue

            target = cv2.resize(
                target,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            target_binary = (
                target > 127
            ).astype(np.uint8)

            iou, dice, precision, recall = calculate_metrics(
                prediction_binary,
                target_binary
            )

            ious.append(iou)
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)

    print()
    print("=" * 60)
    print("U-NET LOCALIZATION EVALUATION")
    print("=" * 60)

    print(f"Evaluated images : {len(ious)}")
    print(f"Skipped images   : {skipped}")

    if len(ious) == 0:
        print()
        print("No images were successfully evaluated.")
        return

    print()
    print(f"Mean IoU         : {np.mean(ious):.4f}")
    print(f"Mean Dice Score  : {np.mean(dices):.4f}")
    print(f"Pixel Precision  : {np.mean(precisions):.4f}")
    print(f"Pixel Recall     : {np.mean(recalls):.4f}")

    print()
    print("Percentages:")
    print(f"IoU              : {np.mean(ious) * 100:.2f}%")
    print(f"Dice             : {np.mean(dices) * 100:.2f}%")
    print(f"Precision        : {np.mean(precisions) * 100:.2f}%")
    print(f"Recall           : {np.mean(recalls) * 100:.2f}%")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()