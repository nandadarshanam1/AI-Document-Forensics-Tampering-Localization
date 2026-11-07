from pathlib import Path
import random
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

AUTHENTIC_DIR = Path("data/authentic")
TAMPERED_DIR = Path("data/tampered")
MASK_DIR = Path("data/masks")

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# FIND DOCUMENTS
# ============================================================

def get_documents():

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    documents = []

    for path in AUTHENTIC_DIR.iterdir():

        if path.is_file() and path.suffix.lower() in valid_extensions:
            documents.append(path)

    return sorted(documents)


# ============================================================
# CREATE EMPTY MASK
# ============================================================

def create_mask(image):

    height = int(image.shape[0])
    width = int(image.shape[1])

    return np.zeros(
        (height, width),
        dtype=np.uint8
    )


# ============================================================
# COPY-MOVE ATTACK
# ============================================================

def copy_move(image):

    result = image.copy()
    mask = create_mask(image)

    height = int(image.shape[0])
    width = int(image.shape[1])

    patch_width = random.randint(
        80,
        max(81, min(300, width // 4))
    )

    patch_height = random.randint(
        40,
        max(41, min(180, height // 8))
    )

    source_x = random.randint(
        20,
        max(21, width - patch_width - 20)
    )

    source_y = random.randint(
        20,
        max(21, height - patch_height - 20)
    )

    destination_x = random.randint(
        20,
        max(21, width - patch_width - 20)
    )

    destination_y = random.randint(
        20,
        max(21, height - patch_height - 20)
    )

    patch = image[
        source_y:source_y + patch_height,
        source_x:source_x + patch_width
    ].copy()

    result[
        destination_y:destination_y + patch_height,
        destination_x:destination_x + patch_width
    ] = patch

    mask[
        destination_y:destination_y + patch_height,
        destination_x:destination_x + patch_width
    ] = 255

    return result, mask


# ============================================================
# ERASE ATTACK
# ============================================================

def erase_region(image):

    result = image.copy()
    mask = create_mask(image)

    height = int(image.shape[0])
    width = int(image.shape[1])

    region_width = random.randint(
        80,
        max(81, width // 5)
    )

    region_height = random.randint(
        30,
        max(31, height // 15)
    )

    x = random.randint(
        50,
        max(51, width - region_width - 50)
    )

    y = random.randint(
        300,
        max(301, height - region_height - 250)
    )

    x2 = min(
        x + region_width,
        width
    )

    y2 = min(
        y + region_height,
        height
    )

    # Estimate nearby background
    sample_x1 = max(0, x - 20)
    sample_x2 = min(width, x + region_width + 20)

    sample_y1 = max(0, y - 20)
    sample_y2 = min(height, y + region_height + 20)

    surrounding = image[
        sample_y1:sample_y2,
        sample_x1:sample_x2
    ]

    background = np.median(
        surrounding,
        axis=(0, 1)
    ).astype(np.uint8)

    result[
        y:y2,
        x:x2
    ] = background

    mask[
        y:y2,
        x:x2
    ] = 255

    return result, mask


# ============================================================
# TEXT REPLACEMENT
# ============================================================

def text_replacement(image):

    result = image.copy()
    mask = create_mask(image)

    height = int(image.shape[0])
    width = int(image.shape[1])

    # Select a realistic text area
    regions = [
        (
            int(width * 0.55),
            int(height * 0.12),
            int(width * 0.92),
            int(height * 0.20)
        ),

        (
            int(width * 0.05),
            int(height * 0.14),
            int(width * 0.48),
            int(height * 0.21)
        ),

        (
            int(width * 0.05),
            int(height * 0.22),
            int(width * 0.50),
            int(height * 0.30)
        )
    ]

    region = random.choice(regions)

    x1 = int(region[0])
    y1 = int(region[1])
    x2 = int(region[2])
    y2 = int(region[3])

    # Cover original text
    result[
        y1:y2,
        x1:x2
    ] = (
        250,
        250,
        250
    )

    replacements = [
        "Invoice No: INV-99999",
        "Invoice No: INV-54321",
        "Date: 31/12/2026",
        "Customer: Modified",
        "GSTIN: 29ZZZZZ9999Z1Z9",
        "Customer: Unknown"
    ]

    replacement = random.choice(
        replacements
    )

    cv2.putText(
        result,
        replacement,
        (
            int(x1 + 5),
            int(y1 + 35)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    mask[
        y1:y2,
        x1:x2
    ] = 255

    return result, mask


# ============================================================
# AMOUNT REPLACEMENT
# ============================================================

def amount_replacement(image):

    result = image.copy()
    mask = create_mask(image)

    height = int(image.shape[0])
    width = int(image.shape[1])

    # Right side of invoice table
    x1 = int(width * 0.78)
    x2 = int(width * 0.95)

    # Pick one table row
    row_height = int(height * 0.043)

    table_start = int(height * 0.36)

    row_number = random.randint(
        1,
        5
    )

    y1 = int(
        table_start +
        row_number * row_height
    )

    y2 = int(
        y1 +
        row_height
    )

    if y2 >= height:
        y2 = height - 5

    result[
        y1:y2,
        x1:x2
    ] = (
        250,
        250,
        250
    )

    fake_amount = random.randint(
        1000,
        999999
    )

    text = f"Rs.{fake_amount:,}"

    cv2.putText(
        result,
        text,
        (
            int(x1 + 5),
            int(y1 + row_height * 0.7)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv2.LINE_AA
    )

    mask[
        y1:y2,
        x1:x2
    ] = 255

    return result, mask


# ============================================================
# SPLICING
# ============================================================

def splicing(image, source_image):

    result = image.copy()
    mask = create_mask(image)

    height = int(image.shape[0])
    width = int(image.shape[1])

    source = cv2.resize(
        source_image,
        (
            width,
            height
        )
    )

    patch_width = random.randint(
        100,
        max(101, width // 5)
    )

    patch_height = random.randint(
        50,
        max(51, height // 10)
    )

    source_x = random.randint(
        0,
        max(0, width - patch_width)
    )

    source_y = random.randint(
        0,
        max(0, height - patch_height)
    )

    destination_x = random.randint(
        0,
        max(0, width - patch_width)
    )

    destination_y = random.randint(
        0,
        max(0, height - patch_height)
    )

    patch = source[
        source_y:source_y + patch_height,
        source_x:source_x + patch_width
    ].copy()

    result[
        destination_y:destination_y + patch_height,
        destination_x:destination_x + patch_width
    ] = patch

    mask[
        destination_y:destination_y + patch_height,
        destination_x:destination_x + patch_width
    ] = 255

    return result, mask


# ============================================================
# GENERATE ONE TAMPERED IMAGE
# ============================================================

def generate_attack(image, source_image):

    attack_number = random.randint(
        1,
        5
    )

    if attack_number == 1:

        return copy_move(image)

    elif attack_number == 2:

        return erase_region(image)

    elif attack_number == 3:

        return text_replacement(image)

    elif attack_number == 4:

        return amount_replacement(image)

    else:

        return splicing(
            image,
            source_image
        )


# ============================================================
# MAIN
# ============================================================

def main():

    TAMPERED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = get_documents()

    print()
    print(
        f"Found {len(documents)} authentic documents."
    )

    if len(documents) == 0:

        print(
            "ERROR: No images found in "
            "data/authentic/"
        )

        return

    successful = 0

    for index, image_path in enumerate(
        documents,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[ERROR] Could not read "
                f"{image_path.name}"
            )

            continue

        # Find a different document for splicing
        source_image = image

        if len(documents) > 1:

            other_documents = [
                p
                for p in documents
                if p != image_path
            ]

            source_path = random.choice(
                other_documents
            )

            source_image = cv2.imread(
                str(source_path)
            )

            if source_image is None:

                source_image = image

        try:

            tampered, mask = generate_attack(
                image,
                source_image
            )

            base_name = image_path.stem

            tampered_path = (
                TAMPERED_DIR /
                f"{base_name}_tampered.png"
            )

            mask_path = (
                MASK_DIR /
                f"{base_name}_mask.png"
            )

            success_image = cv2.imwrite(
                str(tampered_path),
                tampered
            )

            success_mask = cv2.imwrite(
                str(mask_path),
                mask
            )

            if not success_image:
                raise RuntimeError(
                    "Failed to save tampered image"
                )

            if not success_mask:
                raise RuntimeError(
                    "Failed to save mask"
                )

            successful += 1

            if successful % 25 == 0:

                print(
                    f"Generated "
                    f"{successful}/"
                    f"{len(documents)}"
                )

        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )

    print()
    print(
        "=" * 50
    )
    print(
        "DATASET GENERATION COMPLETE"
    )
    print(
        "=" * 50
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Expected: {len(documents)}"
    )

    print(
        f"Tampered directory: "
        f"{TAMPERED_DIR}"
    )

    print(
        f"Mask directory: "
        f"{MASK_DIR}"
    )


if __name__ == "__main__":
    main()