from pathlib import Path
import cv2
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OCR_DIR = Path("results/ocr")
MASK_DIR = Path("results")

OUTPUT_DIR = Path(
    "results/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SELECT DOCUMENT
# ============================================================

document_path = input(
    "Enter original/tampered document path: "
).strip()

document_path = Path(
    document_path
)

if not document_path.exists():

    raise FileNotFoundError(
        f"Document not found: {document_path}"
    )


base_name = document_path.stem

# Example:
# synthetic_invoice_0001_tampered
#
# OCR file:
# synthetic_invoice_0001_tampered_ocr.csv

ocr_file = (
    OCR_DIR /
    f"{base_name}_ocr.csv"
)

if not ocr_file.exists():

    raise FileNotFoundError(
        f"OCR file not found: {ocr_file}"
    )


# ============================================================
# FIND MASK
# ============================================================

# For a tampered image:
#
# synthetic_invoice_0001_tampered.png
#
# corresponding mask:
#
# synthetic_invoice_0001_mask.png

mask_base = base_name.replace(
    "_tampered",
    ""
)

mask_file = (
    MASK_DIR /
    "mask.png"
)

# If a specific dataset mask exists,
# use it instead.

dataset_mask = (
    Path("data/masks") /
    f"{mask_base}_mask.png"
)

if dataset_mask.exists():

    mask_file = dataset_mask

elif not mask_file.exists():

    raise FileNotFoundError(
        "No suitable mask found."
    )


# ============================================================
# LOAD OCR
# ============================================================

ocr_data = pd.read_csv(
    ocr_file
)

required_columns = [
    "text",
    "left",
    "top",
    "width",
    "height"
]

for column in required_columns:

    if column not in ocr_data.columns:

        raise RuntimeError(
            f"Missing OCR column: {column}"
        )


# ============================================================
# LOAD MASK
# ============================================================

mask = cv2.imread(
    str(mask_file),
    cv2.IMREAD_GRAYSCALE
)

if mask is None:

    raise RuntimeError(
        f"Could not read mask: {mask_file}"
    )


mask = (
    mask > 127
).astype("uint8")


# ============================================================
# ANALYZE OCR REGIONS
# ============================================================

results = []

for _, row in ocr_data.iterrows():

    text = str(
        row["text"]
    ).strip()

    if not text:
        continue

    x = int(row["left"])
    y = int(row["top"])

    width = int(row["width"])
    height = int(row["height"])

    x2 = x + width
    y2 = y + height

    # Clamp coordinates to mask
    # boundaries

    x = max(
        0,
        min(x, mask.shape[1])
    )

    y = max(
        0,
        min(y, mask.shape[0])
    )

    x2 = max(
        0,
        min(x2, mask.shape[1])
    )

    y2 = max(
        0,
        min(y2, mask.shape[0])
    )

    if x2 <= x or y2 <= y:

        overlap = 0.0

    else:

        region = mask[
            y:y2,
            x:x2
        ]

        if region.size == 0:

            overlap = 0.0

        else:

            overlap = (
                region.mean()
            )

    confidence = float(
        row.get(
            "conf",
            -1
        )
    )

    results.append(
        {
            "text": text,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "ocr_confidence": confidence,
            "tampering_overlap": overlap
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

analysis = pd.DataFrame(
    results
)


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk(row):

    overlap = row[
        "tampering_overlap"
    ]

    confidence = row[
        "ocr_confidence"
    ]

    # Strong overlap with suspicious
    # pixels

    if overlap >= 0.50:

        return "HIGH"

    elif overlap >= 0.20:

        return "MEDIUM"

    elif confidence >= 80:

        return "LOW"

    else:

        return "UNCERTAIN"


analysis["risk"] = (
    analysis.apply(
        calculate_risk,
        axis=1
    )
)


# ============================================================
# SORT BY SUSPICION
# ============================================================

analysis = analysis.sort_values(
    by=[
        "tampering_overlap",
        "ocr_confidence"
    ],
    ascending=[
        False,
        False
    ]
)


# ============================================================
# SAVE ANALYSIS
# ============================================================

output_file = (
    OUTPUT_DIR /
    f"{base_name}_analysis.csv"
)

analysis.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("DOCUMENT ANALYSIS")
print("=" * 75)

print(
    f"Document: {document_path.name}"
)

print(
    f"Words analyzed: {len(analysis)}"
)

print()

for _, row in analysis.head(20).iterrows():

    print(
        f"{row['risk']:<10} "
        f"{row['text']:<25} "
        f"OCR={row['ocr_confidence']:.1f} "
        f"Overlap="
        f"{row['tampering_overlap'] * 100:.1f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

high_risk = (
    analysis["risk"] == "HIGH"
).sum()

medium_risk = (
    analysis["risk"] == "MEDIUM"
).sum()

uncertain = (
    analysis["risk"] == "UNCERTAIN"
).sum()


print()
print("=" * 75)

print(
    f"HIGH risk regions: {high_risk}"
)

print(
    f"MEDIUM risk regions: {medium_risk}"
)

print(
    f"UNCERTAIN regions: {uncertain}"
)

print()
print(
    f"Analysis saved to:"
)

print(
    output_file
)

print("=" * 75)