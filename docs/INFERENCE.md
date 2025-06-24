# 🔮 Inference Guide

This guide explains how to perform jersey number recognition inference on your own images using our pretrained models.

## Prerequisites

- Python 3.11
- Poetry environment set up (see main README for installation)
- Downloaded model checkpoint and configuration file

## Input Requirements

### Image Directory
The inference script expects a directory containing images. **No specific structure is required** - simply place all your images in a single directory. The script will automatically:

- Scan for supported image formats (PNG, JPG, JPEG - case insensitive)
- Process all found images regardless of filename
- Handle images of any size (they will be automatically resized)

### Example directory structure:
```
your_images/
├── image1.jpg
├── photo_002.png
├── jersey_detection.jpeg
└── any_filename.JPG
```

## Model Setup

### 1. Download Model Checkpoint
Choose a pretrained model from the [Results table](../README.md#-results) in the main README:

- **ViT-S (Small)**: Faster, good performance
- **ViT-B (Base)**: Slower, better performance

Download the checkpoint file (`.pth` format) to your local machine.

### 2. Get Configuration File
The configuration files are included in this repository:
- For ViT-S: [`configs/small16_reid.yaml`](../configs/small16_reid.yaml)
- For ViT-B: [`configs/base8_reid.yaml`](../configs/base8_reid.yaml)

## Usage

### Basic Command
```bash
poetry run python predict_images.py \
    --checkpoint path/to/downloaded/checkpoint.pth \
    --config configs/small16_reid.yaml \
    --image-dir path/to/your/images \
    --output results/predictions
```

### Command Line Arguments
- `--checkpoint`: Path to the downloaded model checkpoint file
- `--config`: Path to the configuration file (use `small16_reid.yaml` for ViT-S or `base8_reid.yaml` for ViT-B)
- `--image-dir`: Directory containing your input images
- `--output`: Output path prefix (without extension)
- `--batch-size` (optional): Override batch size for inference (default from config)

### Example with ViT-B model:
```bash
poetry run python predict_images.py \
    --checkpoint models/vit_b_soccernet.pth \
    --config configs/base8_reid.yaml \
    --image-dir soccer_images/ \
    --output results/match_predictions \
    --batch-size 16
```

## Output Format

The script generates two output files with the same base name but different extensions:

### 1. Detailed Predictions (`predictions.npz`)
NumPy compressed archive containing:
- `image_paths`: List of image filenames (without extension)
- `pred_numbers`: Predicted jersey numbers (0-99)
- `pred_scores`: Confidence scores for top predictions (0-1)
- `uncertainties`: Epistemic uncertainty estimates
- `alphas`: Dirichlet concentration parameters (100-dimensional arrays)
- `probs`: Full probability distributions over all numbers (100-dimensional arrays)
- `widths`, `heights`: Original image dimensions

### 2. Summary CSV (`predictions.csv`)
Simple CSV file with columns:
- `file_name`: Image filename (without extension)
- `pred_number`: Predicted jersey number (0-99)
- `pred_score`: Confidence score (0-1, higher is better)
- `uncertainty`: Uncertainty estimate (lower is better)

### Example CSV output:
```csv
file_name,pred_number,pred_score,uncertainty
image1,10,0.9234,0.0876
photo_002,7,0.8756,0.1234
jersey_detection,23,0.7899,0.2101
```
