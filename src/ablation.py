from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.multimodal_model import MultimodalDocumentModel
from src.models.multimodal_dataset import MultimodalDocumentDataset


# ============================================================
# CONFIGURATION
# ============================================================

AUTHENTIC_DIR = Path("data/authentic")
TAMPERED_DIR = Path("data/tampered")

EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
SEED = 42

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# COLLECT MATCHED DOCUMENTS
# ============================================================

authentic = {}

for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg"
]:

    for path in AUTHENTIC_DIR.glob(
        extension
    ):

        authentic[path.stem] = path


tampered = {}

for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg"
]:

    for path in TAMPERED_DIR.glob(
        extension
    ):

        base = path.stem.replace(
            "_tampered",
            ""
        )

        tampered[base] = path


document_ids = sorted(
    set(authentic.keys())
    &
    set(tampered.keys())
)


print(
    f"Matched document pairs: "
    f"{len(document_ids)}"
)


# ============================================================
# SAME STRICT SPLIT
# ============================================================

random.shuffle(
    document_ids
)

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
    f"Training documents: "
    f"{len(train_ids)}"
)

print(
    f"Validation documents: "
    f"{len(val_ids)}"
)

assert not (
    set(train_ids)
    &
    set(val_ids)
), "DATA LEAKAGE!"


# ============================================================
# BUILD DATASETS
# ============================================================

train_paths = []
train_labels = []

val_paths = []
val_labels = []


for document_id in train_ids:

    train_paths.append(
        authentic[document_id]
    )

    train_labels.append(0)

    train_paths.append(
        tampered[document_id]
    )

    train_labels.append(1)


for document_id in val_ids:

    val_paths.append(
        authentic[document_id]
    )

    val_labels.append(0)

    val_paths.append(
        tampered[document_id]
    )

    val_labels.append(1)


train_dataset = MultimodalDocumentDataset(
    train_paths,
    train_labels
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
# TRAIN ONE VARIANT
# ============================================================

def train_variant(
    name,
    use_ocr,
    use_layout
):

    print()
    print("=" * 70)
    print(
        f"ABLATION: {name}"
    )
    print("=" * 70)

    model = MultimodalDocumentModel()

    model = model.to(device)

    # --------------------------------------------------------
    # Freeze / disable selected modalities
    # --------------------------------------------------------

    if not use_ocr:

        for parameter in (
            model.text_encoder.parameters()
        ):

            parameter.requires_grad = False


    if not use_layout:

        for parameter in (
            model.layout_encoder.parameters()
        ):

            parameter.requires_grad = False


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_accuracy = 0.0


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0.0


        for (
            images,
            text_features,
            layout_features,
            targets
        ) in tqdm(
            train_loader,
            desc=f"{name} Epoch {epoch + 1}/{EPOCHS}"
        ):

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


            # ------------------------------------------------
            # Disable modality by zeroing its input
            # ------------------------------------------------

            if not use_ocr:

                text_features = torch.zeros_like(
                    text_features
                )


            if not use_layout:

                layout_features = torch.zeros_like(
                    layout_features
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
                *
                images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )

            train_correct += (
                predictions == targets
            ).sum().item()

            train_total += (
                targets.size(0)
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

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


                if not use_ocr:

                    text_features = torch.zeros_like(
                        text_features
                    )


                if not use_layout:

                    layout_features = torch.zeros_like(
                        layout_features
                    )


                outputs = model(
                    images,
                    text_features,
                    layout_features
                )


                predictions = outputs.argmax(
                    dim=1
                )


                val_correct += (
                    predictions == targets
                ).sum().item()

                val_total += (
                    targets.size(0)
                )


        train_accuracy = (
            train_correct
            /
            train_total
        )

        val_accuracy = (
            val_correct
            /
            val_total
        )


        print(
            f"Epoch {epoch + 1}: "
            f"Train={train_accuracy * 100:.2f}% "
            f"Val={val_accuracy * 100:.2f}%"
        )


        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "variant":
                        name,

                    "use_ocr":
                        use_ocr,

                    "use_layout":
                        use_layout,

                    "train_document_ids":
                        train_ids,

                    "validation_document_ids":
                        val_ids
                },

                CHECKPOINT_DIR /
                f"ablation_{name}.pth"
            )


    print(
        f"Best {name}: "
        f"{best_accuracy * 100:.2f}%"
    )

    return best_accuracy


# ============================================================
# RUN ABLATIONS
# ============================================================

results = {}


# ------------------------------------------------------------
# Visual + OCR
# ------------------------------------------------------------

results["Visual + OCR"] = train_variant(
    "visual_ocr",
    use_ocr=True,
    use_layout=False
)


# ------------------------------------------------------------
# Visual + Layout
# ------------------------------------------------------------

results["Visual + Layout"] = train_variant(
    "visual_layout",
    use_ocr=False,
    use_layout=True
)


# ------------------------------------------------------------
# Full multimodal
# ------------------------------------------------------------

results["Visual + OCR + Layout"] = train_variant(
    "visual_ocr_layout",
    use_ocr=True,
    use_layout=True
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("ABLATION STUDY COMPLETE")
print("=" * 70)

for name, accuracy in results.items():

    print(
        f"{name:<30} "
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = Path(
    "results/ablation"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    results_dir /
    "ablation_results.txt",
    "w"
) as file:

    file.write(
        "Document Forensics Ablation Study\n"
    )

    file.write(
        "=================================\n\n"
    )

    for name, accuracy in results.items():

        file.write(
            f"{name}: "
            f"{accuracy * 100:.2f}%\n"
        )