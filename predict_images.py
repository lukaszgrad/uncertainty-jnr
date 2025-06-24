#!/usr/bin/env python3
"""
Jersey Number Prediction Script for Image Directories

This script performs inference on a directory of images using a trained jersey
number recognition model. It outputs both detailed predictions (NPZ format) and a
summary CSV file.

Example usage:
    python predict_images.py \
        --checkpoint runs/best_model.pth \
        --image-dir /path/to/images \
        --output results/predictions \
        --batch-size 32

Output files:
    - predictions.npz: Detailed predictions with probabilities, uncertainties, etc.
    - predictions.csv: Simple CSV with file_name, pred_number, pred_score, uncertainty
"""

from time import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import gc
import os

from uncertainty_jnr.data import SimpleImageDataset
from uncertainty_jnr.model import TimmOCRModel
from uncertainty_jnr.augmentation import get_val_transforms
from uncertainty_jnr.utils import load_checkpoint, setup_logging
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for images in a directory"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path (without extension)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size from config"
    )
    args = parser.parse_args()

    # Determine project root directory
    project_root = Path(os.getenv("OCR_DIR", ".")).resolve()

    # Validate image directory
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    # Load config
    config = Config.from_yaml(args.config)

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # Create output directory and setup logging
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, filename="predict_images.log")

    logging.info(f"Using project root: {project_root}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create dataset
    dataset = SimpleImageDataset(
        image_dir=args.image_dir,
        target_size=config.data.target_size,
        transform=get_val_transforms(target_size=config.data.target_size),
    )

    # Check for empty directory
    if len(dataset) == 0:
        logging.error(f"No images found in {args.image_dir}")
        return

    logging.info(f"Found {len(dataset)} images to process")

    # Setup batch size
    batch_size = config.data.val_batch_size
    if args.batch_size is not None:
        batch_size = args.batch_size
        logging.info(f"Using batch size: {batch_size}")

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,  # Don't drop last batch for inference
    )

    # Load model
    model = TimmOCRModel(
        model_name=config.model.model_name,
        pretrained=False,
        classifier_type=config.model.classifier_type,
        embedding_type=config.model.embedding_type,
        per_digit_bias=config.model.per_digit_bias,
        uncertainty_head=config.model.uncertainty_head,
    )
    load_checkpoint(model, checkpoint_path, device, strict=False)
    model = model.to(device)
    model.eval()

    # Initialize predictions storage
    predictions = {
        "image_paths": [],
        "pred_numbers": [],
        "pred_scores": [],
        "uncertainties": [],
        "alphas": [],
        "probs": [],
        "widths": [],
        "heights": [],
    }

    # Generate predictions
    with torch.no_grad():
        start_time = time()
        for batch in tqdm(data_loader, desc="Processing images"):
            images = batch["image"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model_output = model(images)

            # Extract model outputs
            alpha = torch.exp(model_output.number_logits) + 1.0
            probs = model_output.number_probs
            uncertainty = model_output.uncertainty.cpu().numpy()

            # Ensure uncertainty is always 1-D for proper indexing
            if uncertainty.ndim == 0:
                uncertainty = np.array([uncertainty])

            # Get top predictions
            pred_scores, pred_numbers = torch.max(probs, dim=1)
            pred_scores = pred_scores.cpu().numpy()
            pred_numbers = pred_numbers.cpu().numpy()

            # Process each sample in the batch
            for i in range(len(images)):
                predictions["image_paths"].append(batch["image_path"][i])
                predictions["pred_numbers"].append(pred_numbers[i])
                predictions["pred_scores"].append(pred_scores[i])
                predictions["uncertainties"].append(uncertainty[i])
                predictions["alphas"].append(alpha[i].cpu().numpy())
                predictions["probs"].append(probs[i].cpu().numpy())
                predictions["widths"].append(batch["width"][i].item())
                predictions["heights"].append(batch["height"][i].item())

            # Clean up GPU memory
            del model_output, alpha, probs, images, uncertainty
            torch.cuda.empty_cache()

            # Periodic garbage collection
            if len(predictions["image_paths"]) % 500 == 0:
                gc.collect()

        end_time = time()
        total_samples = len(predictions["image_paths"])
        logging.info(
            f"Processed {total_samples} images in {end_time - start_time:.2f}s"
        )
        logging.info(f"Speed: {total_samples / (end_time - start_time):.2f} images/s")

    # Convert to numpy arrays
    pred_arrays = {k: np.array(v) for k, v in predictions.items()}

    # Save NPZ file
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(npz_path, **pred_arrays)
    logging.info(f"Saved detailed predictions to {npz_path}")

    # Create simplified CSV
    csv_data = {
        "file_name": predictions["image_paths"],
        "pred_number": predictions["pred_numbers"],
        "pred_score": predictions["pred_scores"],
        "uncertainty": predictions["uncertainties"],
    }
    csv_df = pd.DataFrame(csv_data)

    csv_path = args.output.with_suffix(".csv")
    csv_df.to_csv(csv_path, index=False)
    logging.info(f"Saved summary predictions to {csv_path}")

    logging.info("Predictions saved to:")
    logging.info(f"  Detailed: {npz_path}")
    logging.info(f"  Summary:  {csv_path}")


if __name__ == "__main__":
    main()
