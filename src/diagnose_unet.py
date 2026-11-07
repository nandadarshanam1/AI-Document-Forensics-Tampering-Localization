from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "tampered"
    / "synthetic_invoice_0001_tampered.png"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_unet.pth"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
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


print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Loading checkpoint...")

model = UNet().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model.eval()

print("Checkpoint loaded.")
print()


print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = image.resize(
    (256, 256)
)

image_np = (
    np.array(image).astype(
        np.float32
    ) / 255.0
)

mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

image_np = (
    image_np - mean
) / std

tensor = torch.from_numpy(
    image_np.transpose(2, 0, 1)
).unsqueeze(0).float().to(device)


print("Running U-Net...")

with torch.no_grad():

    raw_output = model(tensor)

    probability = torch.sigmoid(
        raw_output
    )


raw = raw_output.squeeze().cpu().numpy()

prob = probability.squeeze().cpu().numpy()


print()
print("=" * 60)
print("U-NET RAW OUTPUT DIAGNOSTIC")
print("=" * 60)

print()
print("Raw output:")
print(
    f"Min       : {raw.min():.8f}"
)
print(
    f"Max       : {raw.max():.8f}"
)
print(
    f"Mean      : {raw.mean():.8f}"
)
print(
    f"Std       : {raw.std():.8f}"
)

print()
print("Sigmoid probability:")
print(
    f"Min       : {prob.min():.8f}"
)
print(
    f"Max       : {prob.max():.8f}"
)
print(
    f"Mean      : {prob.mean():.8f}"
)
print(
    f"Std       : {prob.std():.8f}"
)

print()
print("Pixels above thresholds:")

for threshold in [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]:

    count = int(
        (prob >= threshold).sum()
    )

    percentage = (
        count /
        prob.size
    ) * 100

    print(
        f"{threshold:.2f} : "
        f"{count:7d} pixels "
        f"({percentage:.4f}%)"
    )

print()
print("=" * 60)