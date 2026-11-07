from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.multimodal_model import MultimodalDocumentModel
from src.models.multimodal_dataset import MultimodalDocumentDataset


# ============================================================
# CONFIGURATION
# ============================================================

AUTHENTIC_DIR = Path(
    "data/authentic"
)

TAMPERED_DIR = Path(
    "data/tampered"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# COLLECT DATASET
# ============================================================

image_paths = []
labels = []


# Authentic = 0
for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg"
]:

    for path in AUTHENTIC_DIR.glob(
        extension
    ):

        image_paths.append(path)
        labels.append(0)


# Tampered = 1
for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg"
]:

    for path in TAMPERED_DIR.glob(
        extension
    ):

        image_paths.append(path)
        labels.append(1)


# ============================================================
# SHUFFLE
# ============================================================

combined = list(
    zip(
        image_paths,
        labels
    )
)

random.shuffle(
    combined
)

image_paths = [
    x[0]
    for x in combined
]

labels = [
    x[1]
    for x in combined
]


print(
    f"Total documents: "
    f"{len(image_paths)}"
)

print(
    f"Authentic: "
    f"{labels.count(0)}"
)

print(
    f"Tampered: "
    f"{labels.count(1)}"
)


# ============================================================
# DATASET
# ============================================================

dataset = MultimodalDocumentDataset(
    image_paths,
    labels
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


train_dataset, val_dataset = (
    random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=torch.Generator()
        .manual_seed(SEED)
    )
)


print(
    f"Training samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# ============================================================
# DATA LOADERS
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

model = MultimodalDocumentModel()

model = model.to(
    device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )
    )

    for (
        images,
        text_features,
        layout_features,
        targets
    ) in progress:

        images = images.to(
            device,
            non_blocking=True
        )

        text_features = (
            text_features.to(
                device,
                non_blocking=True
            )
        )

        layout_features = (
            layout_features.to(
                device,
                non_blocking=True
            )
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images,
            text_features,
            layout_features
        )

        # Calculate loss
        loss = criterion(
            outputs,
            targets
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        train_correct += (
            predictions == targets
        ).sum().item()

        train_total += (
            targets.size(0)
        )

        progress.set_postfix(
            loss=loss.item()
        )

    train_loss /= train_total

    train_accuracy = (
        train_correct
        / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for (
            images,
            text_features,
            layout_features,
            targets
        ) in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            text_features = (
                text_features.to(
                    device,
                    non_blocking=True
                )
            )

            layout_features = (
                layout_features.to(
                    device,
                    non_blocking=True
                )
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images,
                text_features,
                layout_features
            )

            loss = criterion(
                outputs,
                targets
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions == targets
            ).sum().item()

            val_total += (
                targets.size(0)
            )


    val_loss /= val_total

    val_accuracy = (
        val_correct
        / val_total
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "val_accuracy":
                val_accuracy,

            "epoch":
                epoch + 1
        }

        checkpoint_path = (
            CHECKPOINT_DIR /
            "best_multimodal.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            "✓ Best model saved"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

print(
    "MULTIMODAL TRAINING COMPLETE"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_DIR / 'best_multimodal.pth'}"
)

print("=" * 70)