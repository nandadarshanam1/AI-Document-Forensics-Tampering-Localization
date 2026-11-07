from pathlib import Path

import cv2
import pytesseract
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = (
    TESSERACT_PATH
)


# ============================================================
# SELECT DOCUMENT
# ============================================================

image_path = input(
    "Enter document path: "
).strip()

image_path = Path(
    image_path
)

if not image_path.exists():

    raise FileNotFoundError(
        f"Document not found: {image_path}"
    )


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:

    raise RuntimeError(
        "Could not read document."
    )


# ============================================================
# OCR
# ============================================================

print()
print("Running OCR...")

data = pytesseract.image_to_data(
    image,
    output_type=pytesseract.Output.DATAFRAME
)


# Remove empty OCR results
data = data.dropna(
    subset=["text"]
)

data["text"] = (
    data["text"]
    .astype(str)
    .str.strip()
)

data = data[
    data["text"] != ""
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("OCR RESULTS")
print("=" * 70)

for _, row in data.iterrows():

    text = row["text"]

    x = int(row["left"])
    y = int(row["top"])

    width = int(row["width"])
    height = int(row["height"])

    confidence = float(
        row["conf"]
    )

    print(
        f"Text: {text:<30} "
        f"Position: ({x},{y}) "
        f"Size: {width}x{height} "
        f"Confidence: {confidence:.1f}"
    )


# ============================================================
# SAVE OCR DATA
# ============================================================

output_dir = Path(
    "results/ocr"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_dir /
    f"{image_path.stem}_ocr.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# CREATE OCR VISUALIZATION
# ============================================================

visualization = image.copy()

for _, row in data.iterrows():

    x = int(row["left"])
    y = int(row["top"])

    width = int(row["width"])
    height = int(row["height"])

    confidence = float(
        row["conf"]
    )

    # Ignore extremely low-confidence detections
    if confidence < 20:
        continue

    cv2.rectangle(
        visualization,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )


output_image = (
    output_dir /
    f"{image_path.stem}_ocr_boxes.png"
)

cv2.imwrite(
    str(output_image),
    visualization
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)

print(
    f"Detected words: {len(data)}"
)

print(
    f"OCR CSV: {output_file}"
)

print(
    f"OCR visualization: {output_image}"
)

print("=" * 70)