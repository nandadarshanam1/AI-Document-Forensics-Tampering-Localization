from pathlib import Path
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.models.multimodal_model import MultimodalDocumentModel
from src.models.multimodal_dataset import MultimodalDocumentDataset


# ============================================================
# CONFIGURATION
# ============================================================

AUTHENTIC_DIR = Path("data/authentic")
TAMPERED_DIR = Path("data/tampered")

CHECKPOINT = Path(
    "checkpoints/best_multimodal.pth"
)

BATCH_SIZE = 16
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

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# COLLECT DATA
# ============================================================

image_paths = []
labels = []


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


# Keep the exact same randomization
# used during training.

combined = list(
    zip(
        image_paths,
        labels
    )
)

random.shuffle(combined)

image_paths = [
    item[0]
    for item in combined
]

labels = [
    item[1]
    for item in combined
]


# ============================================================
# DATASET
# ============================================================

dataset = MultimodalDocumentDataset(
    image_paths,
    labels
)


# ============================================================
# SAME 80/20 SPLIT
# ============================================================

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)


_, val_dataset = random_split(
    dataset,
    [
        train_size,
        val_size
    ],
    generator=torch.Generator()
    .manual_seed(SEED)
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = MultimodalDocumentModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print()
print(
    f"Loaded checkpoint: "
    f"{CHECKPOINT}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# INFERENCE
# ============================================================

all_targets = []
all_predictions = []
all_probabilities = []


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

        outputs = model(
            images,
            text_features,
            layout_features
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_targets.extend(
            targets.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

precision = precision_score(
    all_targets,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_targets,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_targets,
    all_predictions,
    zero_division=0
)

auc = roc_auc_score(
    all_targets,
    all_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("MULTIMODAL MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
print(
    classification_report(
        all_targets,
        all_predictions,
        target_names=[
            "Authentic",
            "Tampered"
        ],
        zero_division=0
    )
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = Path(
    "results/evaluation"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    results_dir /
    "multimodal_metrics.txt",
    "w"
) as file:

    file.write(
        "Multimodal Document Forensics\n"
    )

    file.write(
        "=============================\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {auc:.4f}\n"
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

print()
print(
    "Metrics saved to:"
)

print(
    results_dir /
    "multimodal_metrics.txt"
)