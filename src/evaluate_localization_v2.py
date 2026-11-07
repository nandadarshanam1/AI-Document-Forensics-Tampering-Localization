from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "tampered"
MASK_DIR = PROJECT_ROOT / "data" / "masks"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_unet_v2.pth"
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256
THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(
            3,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [
                d4,
                e4
            ],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading U-Net v2...")

model = UNet().to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )

    image = (
        image - mean
    ) / std

    image = torch.from_numpy(
        image.transpose(2, 0, 1)
    )

    return image.float()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = prediction.astype(
        bool
    )

    target = target.astype(
        bool
    )

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    union = np.logical_or(
        prediction,
        target
    ).sum()

    prediction_pixels = prediction.sum()

    target_pixels = target.sum()

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:

        iou = (
            1.0
            if prediction_pixels == 0
            else 0.0
        )

    else:

        iou = (
            intersection /
            union
        )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    denominator = (
        prediction_pixels
        +
        target_pixels
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * intersection
            /
            denominator
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if prediction_pixels == 0:

        precision = (
            1.0
            if target_pixels == 0
            else 0.0
        )

    else:

        precision = (
            intersection /
            prediction_pixels
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if target_pixels == 0:

        recall = (
            1.0
            if prediction_pixels == 0
            else 0.0
        )

    else:

        recall = (
            intersection /
            target_pixels
        )

    return (
        iou,
        dice,
        precision,
        recall
    )


# ============================================================
# DATASET
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob(
        "*_tampered.*"
    )
)

print()
print(
    "Tampered images found:",
    len(image_files)
)


# ============================================================
# EVALUATION
# ============================================================

ious = []
dices = []
precisions = []
recalls = []

skipped = 0


with torch.no_grad():

    progress = tqdm(
        image_files,
        desc="Evaluating U-Net v2"
    )

    for image_path in progress:

        # ----------------------------------------------------
        # Matching mask
        # ----------------------------------------------------

        base_name = image_path.stem.replace(
            "_tampered",
            ""
        )

        mask_path = (
            MASK_DIR
            / f"{base_name}_mask.png"
        )

        if not mask_path.exists():

            skipped += 1

            continue

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception:

            skipped += 1

            continue

        image_tensor = preprocess_image(
            image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        output = model(
            image_tensor
        )

        probability = torch.sigmoid(
            output
        )

        probability = (
            probability
            .squeeze()
            .cpu()
            .numpy()
        )

        prediction = (
            probability >= THRESHOLD
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        target = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if target is None:

            skipped += 1

            continue

        target = cv2.resize(
            target,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        target = (
            target > 127
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        iou, dice, precision, recall = (
            calculate_metrics(
                prediction,
                target
            )
        )

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("U-NET V2 LOCALIZATION EVALUATION")
print("=" * 65)

print()
print(
    f"Evaluated images : {len(ious)}"
)

print(
    f"Skipped images   : {skipped}"
)

if len(ious) == 0:

    print()
    print(
        "ERROR: No images were evaluated."
    )

    raise SystemExit


mean_iou = np.mean(
    ious
)

mean_dice = np.mean(
    dices
)

mean_precision = np.mean(
    precisions
)

mean_recall = np.mean(
    recalls
)


print()
print(
    f"Mean IoU         : {mean_iou:.4f}"
)

print(
    f"Mean Dice Score  : {mean_dice:.4f}"
)

print(
    f"Pixel Precision  : {mean_precision:.4f}"
)

print(
    f"Pixel Recall     : {mean_recall:.4f}"
)


print()
print("Percentages:")

print(
    f"IoU              : "
    f"{mean_iou * 100:.2f}%"
)

print(
    f"Dice             : "
    f"{mean_dice * 100:.2f}%"
)

print(
    f"Precision        : "
    f"{mean_precision * 100:.2f}%"
)

print(
    f"Recall           : "
    f"{mean_recall * 100:.2f}%"
)


# ============================================================
# ADDITIONAL STATISTICS
# ============================================================

print()
print("Distribution:")

print(
    f"Best IoU         : "
    f"{np.max(ious):.4f}"
)

print(
    f"Worst IoU        : "
    f"{np.min(ious):.4f}"
)

print(
    f"Best Dice        : "
    f"{np.max(dices):.4f}"
)

print(
    f"Worst Dice       : "
    f"{np.min(dices):.4f}"
)

print()
print("=" * 65)

print(
    "Evaluation complete."
)