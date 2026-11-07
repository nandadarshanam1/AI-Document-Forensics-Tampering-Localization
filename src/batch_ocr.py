from pathlib import Path
import cv2
import pytesseract
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIRS = [
    Path("data/authentic"),
    Path("data/tampered")
]

OUTPUT_DIR = Path("results/ocr")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# If Windows cannot find tesseract automatically,
# this path ensures the script still works.

TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():

    pytesseract.pytesseract.tesseract_cmd = (
        str(TESSERACT_PATH)
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_paths = []

for directory in DATA_DIRS:

    if not directory.exists():

        print(
            f"Warning: {directory} does not exist."
        )

        continue

    for extension in [
        "*.png",
        "*.jpg",
        "*.jpeg"
    ]:

        image_paths.extend(
            directory.glob(extension)
        )


# Remove duplicates
image_paths = sorted(
    set(image_paths)
)


print("=" * 70)
print("BATCH OCR PROCESSING")
print("=" * 70)

print(
    f"Documents found: {len(image_paths)}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

successful = 0
failed = 0
skipped = 0


for image_path in tqdm(
    image_paths,
    desc="Running OCR"
):

    output_file = (
        OUTPUT_DIR /
        f"{image_path.stem}_ocr.csv"
    )

    # --------------------------------------------------------
    # Skip existing OCR files
    # --------------------------------------------------------

    if output_file.exists():

        skipped += 1

        continue

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                "Could not read image"
            )

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DATAFRAME,
            config="--psm 6"
        )

        # ----------------------------------------------------
        # Keep useful columns
        # ----------------------------------------------------

        columns = [
            "text",
            "left",
            "top",
            "width",
            "height",
            "conf"
        ]

        data = data[columns]

        # Remove empty text
        data["text"] = (
            data["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        data = data[
            data["text"] != ""
        ]

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        data.to_csv(
            output_file,
            index=False
        )

        successful += 1

    except Exception as e:

        failed += 1

        print()
        print(
            f"OCR failed: "
            f"{image_path.name}"
        )

        print(
            f"Reason: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("OCR COMPLETE")
print("=" * 70)

print(
    f"Successful: {successful}"
)

print(
    f"Skipped:    {skipped}"
)

print(
    f"Failed:     {failed}"
)

print(
    f"Total:      {len(image_paths)}"
)

print("=" * 70)