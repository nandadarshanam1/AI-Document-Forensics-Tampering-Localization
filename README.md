# AI-Document-Forensics-Tampering-Localization
A multimodal deep learning framework for document authenticity classification and tampering localization using ResNet18, U-Net, OCR (Tesseract), and PyTorch.
# AI-Based Document Forensics and Tampering Localization Using Deep Learning

A multimodal deep learning framework designed to detect document manipulation, classify document authenticity, and localize potentially tampered regions using Computer Vision, OCR, and Deep Learning.

## Overview

Digital document manipulation has become increasingly sophisticated, making document authenticity verification an important challenge.

This project explores a deep learning-based approach to identify manipulated documents and localize suspicious regions through a combination of visual feature extraction, image segmentation, and textual analysis.

The framework integrates image classification, pixel-level localization, and OCR-based text extraction.

## Key Features

- Document authenticity classification.
- Image-based tampering detection.
- Pixel-level tampering localization.
- OCR-based text extraction using Tesseract.
- Multimodal feature integration.
- Deep learning model training and evaluation.
- GPU-accelerated model training.

## Model Architecture

The project explores three major deep learning components:

### 1. ResNet18
Used for extracting visual features and classifying documents as authentic or tampered.

### 2. U-Net
Used for image segmentation and localization of potentially manipulated document regions.

### 3. Multimodal Deep Learning Model
Combines visual and textual information to improve document analysis and authenticity detection.

### 4. OCR Integration
Tesseract OCR extracts textual information from document images for additional analysis.

## Dataset

The dataset consists of 1,006 document images.

| Category | Images |
|---|---:|
| Authentic Documents | 503 |
| Tampered Documents | 503 |
| Total | 1,006 |
| Training Set | 804 |
| Validation Set | 202 |

The dataset contains authentic and manipulated document samples for model training and evaluation.

## Experimental Results

The multimodal model achieved the following validation performance:

| Metric | Result |
|---|---:|
| Best Validation Accuracy | 89.60% |
| Training Samples | 804 |
| Validation Samples | 202 |

*Note: The reported accuracy represents validation accuracy, not independent test-set accuracy.*

## Technology Stack

- Python
- PyTorch
- Torchvision
- ResNet18
- U-Net
- OpenCV
- Tesseract OCR
- NumPy
- Computer Vision
- Deep Learning

## Hardware

Model training was performed using:

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU.

## Project Structure

```text
AI-Document-Forensics/
│
├── src/
│   ├── classification/
│   ├── localization/
│   ├── multimodal/
│   ├── preprocessing/
│   └── ocr/
│
├── dataset/
├── checkpoints/
├── results/
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

*This represents the proposed repository organisation. Actual directories and filenames may differ.*

## Applications

Potential applications include:

- Digital document verification.
- Invoice manipulation detection.
- Financial document analysis.
- Identity document verification.
- Digital forensic investigations.

## Future Enhancements

- Improve tampering localization accuracy.
- Evaluate performance on independent datasets.
- Support additional document formats.
- Improve robustness against different manipulation techniques.
- Develop a user-friendly document verification interface.

## Project Status

Research and development project. Model training and initial evaluation have been completed.

## Author

**Nanda Darshanam**

B.Tech - Computer Science and Engineering (Data Science)

Dayananda Sagar University, Bengaluru.

## License

This project is licensed under the MIT License.
