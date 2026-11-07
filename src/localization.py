import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("data/tampered")
MASK_DIR = Path("data/masks")

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

class TamperingDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = []

        for image in sorted(self.image_dir.iterdir()):

            # Only image files
            if image.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            # Example:
            # invoice_002_tampered.png
            #
            # becomes:
            # invoice_002_mask.png

            base_name = image.stem.replace(
                "_tampered",
                ""
            )

            mask_name = (
                base_name +
                "_mask.png"
            )

            mask_path = (
                self.mask_dir /
                mask_name
            )

            if mask_path.exists():
                self.images.append(image)

        print(
            "Images with masks:",
            len(self.images)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # Convert:
        # invoice_002_tampered
        #
        # into:
        # invoice_002

        base_name = image_path.stem.replace(
            "_tampered",
            ""
        )

        mask_path = (
            self.mask_dir /
            (
                base_name +
                "_mask.png"
            )
        )

        # Read image
        image = cv2.imread(
            str(image_path)
        )

        # Read mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize image
        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        # Resize mask
        mask = cv2.resize(
            mask,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image
        image = (
            image.astype(np.float32)
            / 255.0
        )

        # Convert mask to binary
        mask = (
            mask.astype(np.float32)
            / 255.0
        )

        mask = (
            mask > 0.5
        ).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Add channel dimension
        # H,W -> 1,H,W
        mask = np.expand_dims(
            mask,
            axis=0
        )

        return (
            torch.tensor(
                image,
                dtype=torch.float32
            ),
            torch.tensor(
                mask,
                dtype=torch.float32
            )
        )


# ============================================================
# U-NET BUILDING BLOCK
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
                kernel_size=3,
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
                kernel_size=3,
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


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
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

        # Pooling
        self.pool = nn.MaxPool2d(
            2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # Final prediction
        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------

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

        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(
            self.pool(e4)
        )

        # -------------------------
        # Decoder
        # -------------------------

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # Final 1-channel mask
        return self.final(d1)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    predictions,
    targets
):

    predictions = torch.sigmoid(
        predictions
    )

    smooth = 1e-6

    intersection = (
        predictions * targets
    ).sum(
        dim=(1, 2, 3)
    )

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum(
            dim=(1, 2, 3)
        )
        +
        targets.sum(
            dim=(1, 2, 3)
        )
        +
        smooth
    )

    return (
        1 - dice
    ).mean()


# ============================================================
# DATASET
# ============================================================

dataset = TamperingDataset(
    IMAGE_DIR,
    MASK_DIR
)

if len(dataset) == 0:

    raise RuntimeError(
        "No matching tampered images and masks found."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

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
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
# MODEL
# ============================================================

model = UNet().to(
    DEVICE
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

best_dice = 0.0


for epoch in range(
    EPOCHS
):

    # -------------------------
    # Training mode
    # -------------------------

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )
    )

    for images, masks in progress:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images
        )

        # Combined BCE + Dice loss
        loss = (
            0.5 *
            bce_loss(
                outputs,
                masks
            )
            +
            0.5 *
            dice_loss(
                outputs,
                masks
            )
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                torch.sigmoid(
                    outputs
                )
                > 0.5
            ).float()

            intersection = (
                predictions * masks
            ).sum(
                dim=(1, 2, 3)
            )

            dice = (
                2 * intersection
                + 1e-6
            ) / (
                predictions.sum(
                    dim=(1, 2, 3)
                )
                +
                masks.sum(
                    dim=(1, 2, 3)
                )
                +
                1e-6
            )

            total_dice += (
                dice.sum().item()
            )

            total_samples += (
                images.size(0)
            )


    average_dice = (
        total_dice /
        total_samples
    )

    average_train_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss = "
        f"{average_train_loss:.4f}, "
        f"Validation Dice = "
        f"{average_dice:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if average_dice > best_dice:

        best_dice = average_dice

        torch.save(
            model.state_dict(),
            "checkpoints/best_unet.pth"
        )

        print(
            f"Saved best U-Net. "
            f"Dice = "
            f"{best_dice:.4f}"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "Localization training complete."
)

print(
    f"Best validation Dice: "
    f"{best_dice:.4f}"
)