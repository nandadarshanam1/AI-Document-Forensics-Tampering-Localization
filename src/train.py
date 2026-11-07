import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# =========================
# Configuration
# =========================

DATA_DIR = Path("classification_data")
BATCH_SIZE = 16
IMAGE_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# Create temporary dataset
# =========================
#
# ImageFolder expects:
#
# dataset/
#   authentic/
#   tampered/
#
# Our folders are currently:
#
# data/authentic
# data/tampered
#
# So we use ImageFolder on data directly.
#

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# Dataset
# =========================

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# =========================
# Train / validation split
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
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

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# Model
# =========================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)


# =========================
# Loss + optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training
# =========================

best_val_accuracy = 0.0

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total

    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Save best model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/best_resnet18.pth"
        )

        print(
            f"Saved best model: "
            f"{best_val_accuracy:.2f}%"
        )


print()
print("Training complete.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)