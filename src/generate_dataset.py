import cv2
import os
import random
import numpy as np

AUTHENTIC_DIR = "data/authentic"
TAMPERED_DIR = "data/tampered"
MASK_DIR = "data/masks"

os.makedirs(TAMPERED_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


def copy_move(image):
    h, w = image.shape[:2]

    patch_w = max(30, w // 8)
    patch_h = max(20, h // 15)

    x1 = random.randint(0, max(0, w - patch_w - 1))
    y1 = random.randint(0, max(0, h - patch_h - 1))

    x2 = random.randint(0, max(0, w - patch_w - 1))
    y2 = random.randint(0, max(0, h - patch_h - 1))

    patch = image[
        y1:y1 + patch_h,
        x1:x1 + patch_w
    ].copy()

    image[
        y2:y2 + patch_h,
        x2:x2 + patch_w
    ] = patch

    mask = np.zeros((h, w), dtype=np.uint8)

    mask[
        y2:y2 + patch_h,
        x2:x2 + patch_w
    ] = 255

    return image, mask


def erase_region(image):
    h, w = image.shape[:2]

    rw = max(30, w // 10)
    rh = max(20, h // 15)

    x = random.randint(0, max(0, w - rw - 1))
    y = random.randint(0, max(0, h - rh - 1))

    tampered = image.copy()

    region = tampered[
        y:y + rh,
        x:x + rw
    ]

    mean_color = region.mean(axis=(0, 1))

    tampered[
        y:y + rh,
        x:x + rw
    ] = mean_color

    mask = np.zeros((h, w), dtype=np.uint8)

    mask[
        y:y + rh,
        x:x + rw
    ] = 255

    return tampered, mask


def process_image(filename):

    path = os.path.join(
        AUTHENTIC_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:
        print("Could not read:", filename)
        return

    manipulation = random.choice([
        copy_move,
        erase_region
    ])

    tampered, mask = manipulation(image.copy())

    base_name = os.path.splitext(filename)[0]

    tampered_name = base_name + "_tampered.png"
    mask_name = base_name + "_mask.png"

    cv2.imwrite(
        os.path.join(TAMPERED_DIR, tampered_name),
        tampered
    )

    cv2.imwrite(
        os.path.join(MASK_DIR, mask_name),
        mask
    )

    print("Created:", tampered_name)


def main():

    files = os.listdir(AUTHENTIC_DIR)

    image_files = [
        f for f in files
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not image_files:
        print("No images found in data/authentic/")
        return

    print(f"Found {len(image_files)} authentic documents.")

    for filename in image_files:
        process_image(filename)

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()