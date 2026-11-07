from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pytesseract
from pytesseract import Output


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_resnet18.pth"
UNET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_unet.pth"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# RESNET18
# ============================================================

from torchvision import models


def load_resnet():

    print()
    print("Loading ResNet18...")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        RESNET_CHECKPOINT,
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

    model = model.to(device)
    model.eval()

    print("Classifier loaded.")

    return model


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

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

        return self.final(d1)


def load_unet():

    print()
    print("Loading U-Net...")

    model = UNet()

    checkpoint = torch.load(
        UNET_CHECKPOINT,
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

    model = model.to(device)
    model.eval()

    print("U-Net loaded.")

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    resized = image.resize(
        (256, 256)
    )

    image_np = np.array(
        resized
    ).astype(
        np.float32
    ) / 255.0

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
    )

    tensor = tensor.unsqueeze(0)

    return tensor.float()


# ============================================================
# OCR
# ============================================================

def run_ocr(image):

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT
    )

    words = []

    confidences = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(
                data["conf"][i]
            )
        except:
            confidence = -1

        if text and confidence >= 0:

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            words.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                }
            )

            confidences.append(
                confidence
            )

    if confidences:

        avg_confidence = float(
            np.mean(confidences)
        )

    else:

        avg_confidence = 0.0

    return words, avg_confidence


# ============================================================
# U-NET LOCALIZATION
# ============================================================

def predict_mask(model, image):

    tensor = preprocess_image(
        image
    ).to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability.squeeze()
        .cpu()
        .numpy()
    )

    binary = (
        probability >= 0.5
    ).astype(
        np.uint8
    )

    return probability, binary


# ============================================================
# MAIN
# ============================================================

def main():

    resnet = load_resnet()

    unet = load_unet()

    print()

    path = input(
        "Enter path to document: "
    ).strip().strip('"')

    image_path = Path(path)

    if not image_path.is_absolute():

        image_path = (
            PROJECT_ROOT /
            image_path
        )

    if not image_path.exists():

        print()
        print(
            "ERROR: File not found:"
        )

        print(image_path)

        return

    print()
    print(
        "Analyzing:",
        image_path
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    original_width, original_height = image.size

    print(
        "Image size:",
        original_width,
        "x",
        original_height
    )

    # --------------------------------------------------------
    # ResNet classification
    # --------------------------------------------------------

    tensor = preprocess_image(
        image
    ).to(device)

    with torch.no_grad():

        output = resnet(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    authentic_probability = float(
        probabilities[0, 0].item()
    )

    tampered_probability = float(
        probabilities[0, 1].item()
    )

    print()
    print(
        f"Visual tampering probability: "
        f"{tampered_probability * 100:.2f}%"
    )

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    probability_mask, binary_mask = predict_mask(
        unet,
        image
    )

    suspicious_pixels = int(
        binary_mask.sum()
    )

    total_pixels = int(
        binary_mask.size
    )

    suspicious_area = (
        suspicious_pixels /
        total_pixels
    ) * 100

    print(
        f"Suspicious pixel area: "
        f"{suspicious_area:.2f}%"
    )

    # --------------------------------------------------------
    # Resize mask to original image
    # --------------------------------------------------------

    mask_original = cv2.resize(
        binary_mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    probability_original = cv2.resize(
        probability_mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Save predicted mask
    # --------------------------------------------------------

    stem = image_path.stem

    mask_path = (
        RESULTS_DIR /
        f"{stem}_predicted_mask.png"
    )

    mask_visual = (
        mask_original * 255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(mask_path),
        mask_visual
    )

    # --------------------------------------------------------
    # Create highlighted image
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    overlay = original_bgr.copy()

    red_layer = np.zeros_like(
        overlay
    )

    red_layer[:, :, 2] = 255

    blended = cv2.addWeighted(
        overlay,
        0.6,
        red_layer,
        0.4,
        0
    )

    mask_bool = (
        mask_original > 0
    )

    overlay[mask_bool] = (
        blended[mask_bool]
    )

    # --------------------------------------------------------
    # Find connected suspicious regions
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_original,
        connectivity=8
    )

    regions = []

    for label in range(
        1,
        num_labels
    ):

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])

        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < 50:
            continue

        regions.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area
            }
        )

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    words, avg_confidence = run_ocr(
        image
    )

    print(
        f"OCR words detected: "
        f"{len(words)}"
    )

    print(
        f"OCR average confidence: "
        f"{avg_confidence:.2f}%"
    )

    # --------------------------------------------------------
    # OCR / mask overlap
    # --------------------------------------------------------

    high_risk_words = []

    for word in words:

        x = word["x"]
        y = word["y"]

        w = word["w"]
        h = word["h"]

        x2 = min(
            x + w,
            original_width
        )

        y2 = min(
            y + h,
            original_height
        )

        if x >= original_width:
            continue

        if y >= original_height:
            continue

        box_mask = mask_original[
            y:y2,
            x:x2
        ]

        if box_mask.size == 0:
            overlap = 0.0

        else:
            overlap = (
                np.mean(
                    box_mask > 0
                )
            )

        word["tampering_overlap"] = (
            float(overlap)
        )

        if overlap >= 0.30:

            high_risk_words.append(
                word
            )

    # --------------------------------------------------------
    # Save OCR CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR /
        f"{stem}_ocr.csv"
    )

    import csv

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "text",
                "confidence",
                "x",
                "y",
                "w",
                "h",
                "tampering_overlap"
            ]
        )

        for word in words:

            writer.writerow(
                [
                    word["text"],
                    f"{word['confidence']:.2f}",
                    word["x"],
                    word["y"],
                    word["w"],
                    word["h"],
                    f"{word.get('tampering_overlap', 0):.4f}"
                ]
            )

    # --------------------------------------------------------
    # Save highlighted image
    # --------------------------------------------------------

    highlighted_path = (
        RESULTS_DIR /
        f"{stem}_highlighted.png"
    )

    cv2.imwrite(
        str(highlighted_path),
        overlay
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = (
        RESULTS_DIR /
        f"{stem}_report.txt"
    )

    classification = (
        "HIGH TAMPERING RISK"
        if tampered_probability >= 0.75
        else
        "POTENTIAL TAMPERING"
        if tampered_probability >= 0.50
        else
        "LOW TAMPERING RISK"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DOCUMENT FORENSIC ANALYSIS REPORT\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Document: {image_path.name}\n"
        )

        f.write(
            f"Image size: "
            f"{original_width} x "
            f"{original_height}\n\n"
        )

        f.write(
            f"Visual tampering probability: "
            f"{tampered_probability * 100:.2f}%\n"
        )

        f.write(
            f"Classification: "
            f"{classification}\n\n"
        )

        f.write(
            f"Suspicious pixel area: "
            f"{suspicious_area:.2f}%\n"
        )

        f.write(
            f"Suspicious regions: "
            f"{len(regions)}\n\n"
        )

        f.write(
            f"OCR words detected: "
            f"{len(words)}\n"
        )

        f.write(
            f"OCR average confidence: "
            f"{avg_confidence:.2f}%\n"
        )

        f.write(
            f"OCR regions overlapping "
            f"suspicious mask: "
            f"{len(high_risk_words)}\n\n"
        )

        f.write(
            "Detected suspicious regions\n"
        )

        f.write(
            "-" * 50 + "\n"
        )

        for i, region in enumerate(
            regions,
            start=1
        ):

            f.write(
                f"Region {i}: "
                f"x={region['x']}, "
                f"y={region['y']}, "
                f"width={region['w']}, "
                f"height={region['h']}, "
                f"area={region['area']} pixels\n"
            )

        f.write(
            "\nImportant:\n"
        )

        f.write(
            "This system identifies patterns associated "
            "with potential document manipulation. "
            "It should not be treated as definitive proof "
            "of fraud or document authenticity.\n"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Visual tampering probability: "
        f"{tampered_probability * 100:.2f}%"
    )

    print(
        f"Suspicious pixel area: "
        f"{suspicious_area:.2f}%"
    )

    print(
        f"Suspicious regions: "
        f"{len(regions)}"
    )

    print(
        f"OCR words detected: "
        f"{len(words)}"
    )

    print(
        f"OCR average confidence: "
        f"{avg_confidence:.2f}%"
    )

    print()
    print(
        "Classification:",
        classification
    )

    print()
    print(
        "Predicted mask:"
    )

    print(mask_path)

    print()
    print(
        "Highlighted image:"
    )

    print(highlighted_path)

    print()
    print(
        "OCR CSV:"
    )

    print(csv_path)

    print()
    print(
        "Report:"
    )

    print(report_path)


if __name__ == "__main__":
    main()