from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pytesseract

# Configure Tesseract OCR before any pipeline functions are executed.
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_EXE.is_file():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
else:
    # Keep the script importable, but fail clearly when OCR is first used.
    print(
        "WARNING: Tesseract OCR executable was not found at: "
        f"{TESSERACT_EXE}"
    )

from pytesseract import Output
from torchvision import models


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESNET_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_resnet18.pth"
)

MULTIMODAL_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_multimodal_strict.pth"
)

UNET_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_unet_v2.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "forensic"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# RESNET18
# ============================================================

def load_resnet():

    print()
    print("Loading ResNet18...")

    model = models.resnet18(
        weights=None
    )

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

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)
    model.eval()

    print("Classifier loaded.")

    return model


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
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

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
    print("Loading U-Net v2...")

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

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)
    model.eval()

    print("U-Net v2 loaded.")

    return model


# ============================================================
# EXACT MULTIMODAL ARCHITECTURE
# ============================================================

class MultimodalDocumentModel(nn.Module):

    def __init__(
        self,
        text_dim=4,
        layout_dim=4,
        num_classes=2
    ):
        super().__init__()

        backbone = models.resnet18(
            weights=None
        )

        visual_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.visual_encoder = backbone

        self.text_encoder = nn.Sequential(

            nn.Linear(
                text_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()
        )

        self.layout_encoder = nn.Sequential(

            nn.Linear(
                layout_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()
        )

        fusion_dim = (
            visual_dim
            + 128
            + 128
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(
            64,
            num_classes
        )

    def forward(
        self,
        image,
        text_features,
        layout_features
    ):

        visual_features = (
            self.visual_encoder(
                image
            )
        )

        text_features = (
            self.text_encoder(
                text_features
            )
        )

        layout_features = (
            self.layout_encoder(
                layout_features
            )
        )

        fused = torch.cat(
            [
                visual_features,
                text_features,
                layout_features
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        output = self.classifier(
            fused
        )

        return output


def load_multimodal():

    print()
    print("Loading multimodal model...")

    model = MultimodalDocumentModel()

    checkpoint = torch.load(
        MULTIMODAL_CHECKPOINT,
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

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)
    model.eval()

    print("Multimodal model loaded.")

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    size
):

    image = image.resize(
        (
            size,
            size
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

    tensor = torch.from_numpy(
        image.transpose(
            2,
            0,
            1
        )
    )

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

    for i in range(
        len(data["text"])
    ):

        text = data["text"][i].strip()

        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            confidence = -1

        if (
            text
            and confidence >= 0
        ):

            words.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "w": int(data["width"][i]),
                    "h": int(data["height"][i])
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

    return (
        words,
        avg_confidence
    )


# ============================================================
# FEATURES
# ============================================================

def extract_features(
    words,
    width,
    height
):

    if not words:

        return (
            np.zeros(
                4,
                dtype=np.float32
            ),
            np.zeros(
                4,
                dtype=np.float32
            )
        )

    word_count = len(words)

    avg_confidence = np.mean(
        [
            word["confidence"]
            for word in words
        ]
    )

    total_text_area = sum(
        word["w"] * word["h"]
        for word in words
    )

    image_area = width * height

    text_area_ratio = (
        total_text_area /
        image_area
    )

    avg_word_length = np.mean(
        [
            len(word["text"])
            for word in words
        ]
    )

    text_features = np.array(
        [
            word_count / 100.0,
            avg_confidence / 100.0,
            text_area_ratio,
            avg_word_length / 10.0
        ],
        dtype=np.float32
    )

    xs = np.array(
        [
            word["x"] / width
            for word in words
        ],
        dtype=np.float32
    )

    ys = np.array(
        [
            word["y"] / height
            for word in words
        ],
        dtype=np.float32
    )

    layout_features = np.array(
        [
            np.mean(xs),
            np.mean(ys),
            np.std(xs),
            np.std(ys)
        ],
        dtype=np.float32
    )

    return (
        text_features,
        layout_features
    )


# ============================================================
# U-NET PREDICTION
# ============================================================

def predict_unet(
    model,
    image
):

    tensor = preprocess_image(
        image,
        256
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(
            tensor
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

    binary = (
        probability >= 0.5
    ).astype(
        np.uint8
    )

    return (
        probability,
        binary
    )


# ============================================================
# REGIONS
# ============================================================

def find_regions(mask):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    regions = []

    for label in range(
        1,
        num_labels
    ):

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
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

    return regions


# ============================================================
# MAIN
# ============================================================

def main():

    resnet = load_resnet()

    multimodal = load_multimodal()

    unet = load_unet()

    print()

    # Accept a document path passed by Streamlit or from the command line.
    # If no argument is provided, prompt for a path interactively.
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1].strip().strip('"')
    else:
        path = input(
            "Enter path to document: "
        ).strip().strip('"')

    image_path = Path(path).expanduser()

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_path = image_path.resolve()

    if not image_path.is_file():
        print()
        print("ERROR: Document file not found or path is not a file:")
        print(image_path)
        return

    try:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
    except Exception as exc:
        print()
        print("ERROR: Could not open document as an image:")
        print(image_path)
        print(f"Details: {exc}")
        return

    width, height = image.size

    print()
    print("Analyzing:", image_path)
    print("Image size:", width, "x", height)

    # ========================================================
    # RESNET
    # ========================================================

    resnet_tensor = preprocess_image(
        image,
        224
    ).unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        output = resnet(
            resnet_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

    visual_risk = float(
        probabilities[
            0,
            1
        ].item()
    )

    print()
    print(
        "Visual tampering probability:",
        f"{visual_risk * 100:.2f}%"
    )

    # ========================================================
    # OCR
    # ========================================================

    words, ocr_confidence = run_ocr(
        image
    )

    print(
        "OCR words detected:",
        len(words)
    )

    print(
        "OCR average confidence:",
        f"{ocr_confidence:.2f}%"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    text_features, layout_features = (
        extract_features(
            words,
            width,
            height
        )
    )

    text_tensor = torch.from_numpy(
        text_features
    ).unsqueeze(
        0
    ).to(device)

    layout_tensor = torch.from_numpy(
        layout_features
    ).unsqueeze(
        0
    ).to(device)

    # ========================================================
    # MULTIMODAL
    # ========================================================

    with torch.no_grad():

        multimodal_output = multimodal(
            resnet_tensor,
            text_tensor,
            layout_tensor
        )

        multimodal_probabilities = (
            torch.softmax(
                multimodal_output,
                dim=1
            )
        )

    multimodal_risk = float(
        multimodal_probabilities[
            0,
            1
        ].item()
    )

    print(
        "Multimodal tampering probability:",
        f"{multimodal_risk * 100:.2f}%"
    )

    # ========================================================
    # U-NET
    # ========================================================

    probability_mask, binary_mask = (
        predict_unet(
            unet,
            image
        )
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
    ) * 100.0

    predicted_mask = cv2.resize(
        binary_mask,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    print(
        "Suspicious pixel area:",
        f"{suspicious_area:.2f}%"
    )

    regions = find_regions(
        predicted_mask
    )

    print(
        "Suspicious regions:",
        len(regions)
    )

    # ========================================================
    # OCR / MASK OVERLAP
    # ========================================================

    high_risk_words = []

    for word in words:

        x = word["x"]
        y = word["y"]

        w = word["w"]
        h = word["h"]

        x2 = min(
            x + w,
            width
        )

        y2 = min(
            y + h,
            height
        )

        if (
            x >= width
            or y >= height
            or x2 <= x
            or y2 <= y
        ):

            overlap = 0.0

        else:

            box_mask = predicted_mask[
                y:y2,
                x:x2
            ]

            if box_mask.size == 0:

                overlap = 0.0

            else:

                overlap = float(
                    np.mean(
                        box_mask > 0
                    )
                )

        word[
            "tampering_overlap"
        ] = overlap

        if overlap >= 0.30:

            high_risk_words.append(
                word
            )

    # ========================================================
    # COMBINED RISK
    # ========================================================

    combined_risk = (
        0.4 * visual_risk
        +
        0.6 * multimodal_risk
    )

    if combined_risk >= 0.75:

        classification = (
            "HIGH TAMPERING RISK"
        )

    elif combined_risk >= 0.50:

        classification = (
            "POTENTIAL TAMPERING"
        )

    else:

        classification = (
            "LOW TAMPERING RISK"
        )

    print()
    print(
        "Combined tampering risk:",
        f"{combined_risk * 100:.2f}%"
    )

    print()
    print(
        "Classification:",
        classification
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    stem = image_path.stem

    mask_path = (
        RESULTS_DIR
        / f"{stem}_predicted_mask.png"
    )

    highlighted_path = (
        RESULTS_DIR
        / f"{stem}_highlighted.png"
    )

    ocr_path = (
        RESULTS_DIR
        / f"{stem}_ocr.csv"
    )

    report_path = (
        RESULTS_DIR
        / f"{stem}_report.txt"
    )

    # ========================================================
    # SAVE MASK
    # ========================================================

    mask_image = (
        predicted_mask * 255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(mask_path),
        mask_image
    )

    # ========================================================
    # HIGHLIGHT
    # ========================================================

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
        0.60,
        red_layer,
        0.40,
        0
    )

    mask_bool = (
        predicted_mask > 0
    )

    overlay[mask_bool] = (
        blended[mask_bool]
    )

    for index, region in enumerate(
        regions,
        start=1
    ):

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        cv2.rectangle(
            overlay,
            (x, y),
            (
                x + w,
                y + h
            ),
            (0, 0, 255),
            2
        )

        cv2.putText(
            overlay,
            f"Region {index}",
            (
                x,
                max(
                    y - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(
        str(highlighted_path),
        overlay
    )

    # ========================================================
    # OCR CSV
    # ========================================================

    with open(
        ocr_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )

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
                    f"{word.get('tampering_overlap', 0.0):.4f}"
                ]
            )

    # ========================================================
    # REPORT
    # ========================================================

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DOCUMENT FORENSIC ANALYSIS REPORT\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Document: {image_path.name}\n"
        )

        f.write(
            f"Image size: {width} x {height}\n\n"
        )

        f.write(
            "CLASSIFICATION\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            f"{classification}\n\n"
        )

        f.write(
            "MODEL RESULTS\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            f"Visual tampering probability: "
            f"{visual_risk * 100:.2f}%\n"
        )

        f.write(
            f"Multimodal tampering probability: "
            f"{multimodal_risk * 100:.2f}%\n"
        )

        f.write(
            f"Combined tampering risk: "
            f"{combined_risk * 100:.2f}%\n\n"
        )

        f.write(
            "LOCALIZATION\n"
        )

        f.write(
            "-" * 60
            + "\n"
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
            "OCR ANALYSIS\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            f"OCR words detected: "
            f"{len(words)}\n"
        )

        f.write(
            f"OCR average confidence: "
            f"{ocr_confidence:.2f}%\n"
        )

        f.write(
            f"OCR regions overlapping "
            f"suspicious mask: "
            f"{len(high_risk_words)}\n\n"
        )

        f.write(
            "SUSPICIOUS REGIONS\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        if regions:

            for index, region in enumerate(
                regions,
                start=1
            ):

                f.write(
                    f"Region {index}: "
                    f"x={region['x']}, "
                    f"y={region['y']}, "
                    f"width={region['w']}, "
                    f"height={region['h']}, "
                    f"area={region['area']} pixels\n"
                )

        else:

            f.write(
                "No suspicious connected regions detected.\n"
            )

        f.write(
            "\nDISCLAIMER\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(
            "This system identifies patterns associated "
            "with potential document manipulation. "
            "It should not be treated as definitive "
            "proof of fraud or document authenticity.\n"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 65)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 65)

    print()
    print(
        f"Visual tampering probability: "
        f"{visual_risk * 100:.2f}%"
    )

    print(
        f"Multimodal tampering probability: "
        f"{multimodal_risk * 100:.2f}%"
    )

    print(
        f"Combined tampering risk: "
        f"{combined_risk * 100:.2f}%"
    )

    print()
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
        f"{ocr_confidence:.2f}%"
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

    print(
        mask_path
    )

    print()
    print(
        "Highlighted image:"
    )

    print(
        highlighted_path
    )

    print()
    print(
        "OCR CSV:"
    )

    print(
        ocr_path
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()