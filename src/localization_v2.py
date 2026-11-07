from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "tampered"
MASK_DIR = PROJECT_ROOT / "data" / "masks"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT = CHECKPOINT_DIR / "best_unet_v2.pth"


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256

BATCH_SIZE = 8

EPOCHS = 15

LEARNING_RATE = 0.0003

THRESHOLD = 0.5

SEED = 42


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
# DATASET
# ============================================================

class TamperingDataset(Dataset):

    def __init__(
        self,
        image_files
    ):

        self.image_files = image_files

    def __len__(self):

        return len(
            self.image_files
        )

    def __getitem__(
        self,
        index
    ):

        image_path = self.image_files[index]

        # ----------------------------------------------------
        # Find matching mask
        # ----------------------------------------------------

        base_name = image_path.stem.replace(
            "_tampered",
            ""
        )

        mask_path = (
            MASK_DIR /
            f"{base_name}_mask.png"
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

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

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

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
        ).float()

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.resize(
            mask,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


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

    def forward(
        self,
        x
    ):

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

        self.pool = nn.MaxPool2d(
            2
        )

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

    def forward(
        self,
        x
    ):

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
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    smooth=1.0
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.view(
        probabilities.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + smooth
    ) / (
        probabilities.sum(dim=1)
        +
        targets.sum(dim=1)
        +
        smooth
    )

    return (
        1 - dice
    ).mean()


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    logits,
    targets
):

    # --------------------------------------------------------
    # Positive pixel weighting
    # --------------------------------------------------------

    positive_pixels = targets.sum()

    negative_pixels = (
        targets.numel()
        - positive_pixels
    )

    if positive_pixels > 0:

        pos_weight = (
            negative_pixels /
            positive_pixels
        )

        # Prevent extreme weighting
        pos_weight = torch.clamp(
            pos_weight,
            min=1.0,
            max=20.0
        )

    else:

        pos_weight = torch.tensor(
            1.0,
            device=logits.device
        )

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight
    )

    dice = dice_loss(
        logits,
        targets
    )

    return (
        0.5 * bce
        +
        0.5 * dice
    )


# ============================================================
# DATA
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob(
        "*_tampered.*"
    )
)

print()
print(
    "Images with masks:",
    len(image_files)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

dataset = TamperingDataset(
    image_files
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_dataset, val_dataset = random_split(
    dataset,
    [
        train_size,
        val_size
    ],
    generator=torch.Generator().manual_seed(
        SEED
    )
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

model = UNet().to(
    device
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_dice = -1.0


for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_dice_total = 0.0

    train_batches = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, masks in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Training Dice
        # ----------------------------------------------------

        with torch.no_grad():

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            intersection = (
                predictions * masks
            ).sum(
                dim=(1, 2, 3)
            )

            denominator = (
                predictions.sum(
                    dim=(1, 2, 3)
                )
                +
                masks.sum(
                    dim=(1, 2, 3)
                )
            )

            dice = (
                (
                    2 * intersection + 1
                )
                /
                (
                    denominator + 1
                )
            ).mean()

        train_loss += loss.item()

        train_dice_total += dice.item()

        train_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{dice.item():.4f}"
        )

    train_loss /= train_batches

    train_dice = (
        train_dice_total /
        train_batches
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_dice_total = 0.0

    val_iou_total = 0.0

    val_batches = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                masks
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            intersection = (
                predictions * masks
            ).sum(
                dim=(1, 2, 3)
            )

            pred_sum = predictions.sum(
                dim=(1, 2, 3)
            )

            mask_sum = masks.sum(
                dim=(1, 2, 3)
            )

            union = (
                pred_sum
                +
                mask_sum
                -
                intersection
            )

            dice = (
                (
                    2 * intersection + 1
                )
                /
                (
                    pred_sum
                    +
                    mask_sum
                    +
                    1
                )
            ).mean()

            iou = (
                (
                    intersection + 1
                )
                /
                (
                    union + 1
                )
            ).mean()

            val_loss += loss.item()

            val_dice_total += dice.item()

            val_iou_total += iou.item()

            val_batches += 1

    val_loss /= val_batches

    val_dice = (
        val_dice_total /
        val_batches
    )

    val_iou = (
        val_iou_total /
        val_batches
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss {train_loss:.4f} "
        f"Train Dice {train_dice:.4f} "
        f"Val Loss {val_loss:.4f} "
        f"Val Dice {val_dice:.4f} "
        f"Val IoU {val_iou:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            CHECKPOINT
        )

        print(
            f"Saved best model. "
            f"Validation Dice: "
            f"{best_dice:.4f}"
        )


print()
print("=" * 60)

print(
    "Training complete."
)

print(
    f"Best validation Dice: "
    f"{best_dice:.4f}"
)

print(
    "Checkpoint:"
)

print(
    CHECKPOINT
)

print("=" * 60)