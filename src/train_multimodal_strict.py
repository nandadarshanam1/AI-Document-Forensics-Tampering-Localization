from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.multimodal_model import MultimodalDocumentModel
from src.models.multimodal_dataset import MultimodalDocumentDataset


# ============================================================
# CONFIGURATION
# ============================================================

AUTHENTIC_DIR = Path("data/authentic")
TAMPERED_DIR = Path("data/tampered")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# COLLECT DOCUMENT PAIRS
# ============================================================

authentic = {}

for extension in ["*.png", "*.jpg", "*.jpeg"]:

    for path in AUTHENTIC_DIR.glob(extension):

        authentic[path.stem] = path


tampered = {}

for extension in ["*.png", "*.jpg", "*.jpeg"]:

    for path in TAMPERED_DIR.glob(extension):

        # synthetic_invoice_0001_tampered
        # becomes synthetic_invoice_0001

        base = path.stem.replace(
            "_tampered",
            ""
        )

        tampered[base] = path


# ============================================================
# FIND MATCHED DOCUMENTS
# ============================================================

document_ids = sorted(
    set(authentic.keys())
    & set(tampered.keys())
)


print(
    f"Matched document pairs: "
    f"{len(document_ids)}"
)


if len(document_ids) == 0:

    raise RuntimeError(
        "No authentic/tampered document pairs found."
    )


# ============================================================
# DOCUMENT-LEVEL SPLIT
# ============================================================

random.shuffle(document_ids)

split_index = int(
    0.8 * len(document_ids)
)

train_ids = document_ids[
    :split_index
]

val_ids = document_ids[
    split_index:
]


print(
    f"Training document IDs: "
    f"{len(train_ids)}"
)

print(
    f"Validation document IDs: "
    f"{len(val_ids)}"
)


# ============================================================
# VERIFY NO OVERLAP
# ============================================================

overlap = (
    set(train_ids)
    & set(val_ids)
)

print(
    f"Document ID overlap: "
    f"{len(overlap)}"
)

if overlap:

    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )


# ============================================================
# BUILD IMAGE LIST
# ============================================================

image_paths = []
labels = []

# Training:
# authentic + tampered version
# of each training document

for document_id in train_ids:

    image_paths.append(
        authentic[document_id]
    )

    labels.append(0)

    image_paths.append(
        tampered[document_id]
    )

    labels.append(1)


# Validation dataset separately

val_paths = []
val_labels = []

for document_id in val_ids:

    val_paths.append(
        authentic[document_id]
    )

    val_labels.append(0)

    val_paths.append(
        tampered[document_id]
    )

    val_labels.append(1)


print(
    f"Training images: "
    f"{len(image_paths)}"
)

print(
    f"Validation images: "
    f"{len(val_paths)}"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = MultimodalDocumentDataset(
    image_paths,
    labels
)

val_dataset = MultimodalDocumentDataset(
    val_paths,
    val_labels
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

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
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

        text_features = text_features.to(
            device,
            non_blocking=True
        )

        layout_features = layout_features.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images,
            text_features,
            layout_features
        )

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == targets
        ).sum().item()

        train_total += targets.size(0)

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

            text_features = text_features.to(
                device,
                non_blocking=True
            )

            layout_features = layout_features.to(
                device,
                non_blocking=True
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

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == targets
            ).sum().item()

            val_total += targets.size(0)


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
    # SAVE BEST
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1,

                "train_document_ids":
                    train_ids,

                "validation_document_ids":
                    val_ids
            },
            CHECKPOINT_DIR /
            "best_multimodal_strict.pth"
        )

        print(
            "✓ Best strict model saved"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

print(
    "STRICT MULTIMODAL TRAINING COMPLETE"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Checkpoint:"
)

print(
    CHECKPOINT_DIR /
    "best_multimodal_strict.pth"
)

print("=" * 70)