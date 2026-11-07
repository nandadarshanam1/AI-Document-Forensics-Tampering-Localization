from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalDocumentDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels
    ):
        self.image_paths = image_paths
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    # ========================================================
    # OCR FEATURES
    # ========================================================

    def get_ocr_features(self, image_path):

        base_name = image_path.stem

        ocr_file = (
            Path("results/ocr") /
            f"{base_name}_ocr.csv"
        )

        # ----------------------------------------------------
        # If OCR file doesn't exist, return zeros
        # ----------------------------------------------------

        if not ocr_file.exists():

            return np.zeros(
                4,
                dtype=np.float32
            )

        try:

            data = pd.read_csv(
                ocr_file
            )

            if len(data) == 0:

                return np.zeros(
                    4,
                    dtype=np.float32
                )

            # Remove empty text
            if "text" in data.columns:

                data = data[
                    data["text"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    != ""
                ]

            # Number of detected words
            word_count = len(data)

            # Average OCR confidence
            if "conf" in data.columns:

                confidence = pd.to_numeric(
                    data["conf"],
                    errors="coerce"
                )

                confidence = confidence[
                    confidence >= 0
                ]

                if len(confidence) > 0:

                    avg_confidence = (
                        confidence.mean()
                        / 100.0
                    )

                else:

                    avg_confidence = 0.0

            else:

                avg_confidence = 0.0

            # Total text area
            if (
                "width" in data.columns
                and
                "height" in data.columns
            ):

                text_area = (
                    data["width"]
                    * data["height"]
                ).sum()

            else:

                text_area = 0

            # Average word length
            if "text" in data.columns:

                word_lengths = (
                    data["text"]
                    .astype(str)
                    .str.len()
                )

                avg_word_length = (
                    word_lengths.mean()
                    if len(word_lengths) > 0
                    else 0
                )

            else:

                avg_word_length = 0

            # Normalize word count
            word_count = min(
                word_count / 100.0,
                1.0
            )

            # Normalize text area
            text_area = min(
                text_area / 500000.0,
                1.0
            )

            # Normalize word length
            avg_word_length = min(
                avg_word_length / 20.0,
                1.0
            )

            features = np.array(
                [
                    word_count,
                    avg_confidence,
                    text_area,
                    avg_word_length
                ],
                dtype=np.float32
            )

            return features

        except Exception as e:

            print(
                f"OCR error for "
                f"{image_path.name}: {e}"
            )

            return np.zeros(
                4,
                dtype=np.float32
            )

    # ========================================================
    # LAYOUT FEATURES
    # ========================================================

    def get_layout_features(self, image_path):

        base_name = image_path.stem

        ocr_file = (
            Path("results/ocr") /
            f"{base_name}_ocr.csv"
        )

        if not ocr_file.exists():

            return np.zeros(
                4,
                dtype=np.float32
            )

        try:

            data = pd.read_csv(
                ocr_file
            )

            if len(data) == 0:

                return np.zeros(
                    4,
                    dtype=np.float32
                )

            # Image dimensions
            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                return np.zeros(
                    4,
                    dtype=np.float32
                )

            image_height, image_width = (
                image.shape[:2]
            )

            # ------------------------------------------------
            # Normalize OCR coordinates
            # ------------------------------------------------

            centers_x = (
                data["left"]
                + data["width"] / 2
            ) / image_width

            centers_y = (
                data["top"]
                + data["height"] / 2
            ) / image_height

            # Average horizontal position
            mean_x = centers_x.mean()

            # Average vertical position
            mean_y = centers_y.mean()

            # Spatial spread
            spread_x = centers_x.std()

            spread_y = centers_y.std()

            features = np.array(
                [
                    mean_x,
                    mean_y,
                    spread_x,
                    spread_y
                ],
                dtype=np.float32
            )

            return features

        except Exception as e:

            print(
                f"Layout error for "
                f"{image_path.name}: {e}"
            )

            return np.zeros(
                4,
                dtype=np.float32
            )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = self.labels[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read "
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = self.transform(
            image
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text_features = (
            self.get_ocr_features(
                image_path
            )
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        layout_features = (
            self.get_layout_features(
                image_path
            )
        )

        return (
            image,
            torch.tensor(
                text_features,
                dtype=torch.float32
            ),
            torch.tensor(
                layout_features,
                dtype=torch.float32
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.image_paths
        )