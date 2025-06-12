import logging
import torch
import random
import numpy as np
from pathlib import Path


def setup_logging(output_dir: Path, filename: str = "log.txt") -> None:
    """Setup logging to file and console.

    Parameters
    ----------
    output_dir : Path
        Directory to save log file
    filename : str
        Name of the log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / filename),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    val_metrics: dict,
    global_step: int,
    run_dir: Path,
    is_best: bool = False,
    save_optimizer: bool = False,
    save_best_only: bool = False,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "global_step": global_step,
    }
    if save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if not save_best_only:
        torch.save(checkpoint, run_dir / "latest_checkpoint.pt")

    # Save best checkpoint if needed
    if is_best:
        torch.save(checkpoint, run_dir / "best_checkpoint.pt")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    """Load checkpoint with proper key handling for compiled models.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load weights into
    checkpoint_path : Path
        Path to checkpoint file
    device : torch.device
        Device to load checkpoint to
    strict : bool
        Whether to use strict loading
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Fix keys for compiled model
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' from keys if present
        if "_orig_mod." in k:
            new_key = k.replace("._orig_mod.", ".")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load the modified state dict with non-strict mode
    try:
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=strict
        )
        if missing_keys:
            # Expected for new components
            logging.info(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
        logging.info("Checkpoint loaded successfully")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise
    return model


def save_predictions(predictions: dict, output_dir: Path) -> None:
    """Save predictions per match."""
    for match_id, match_preds in predictions.items():
        # Convert all tensors to numpy arrays
        match_preds = {
            k: v.cpu().numpy() if torch.is_tensor(v) else v
            for k, v in match_preds.items()
        }

        # Save as npz file
        output_path = output_dir / f"{match_id}_predictions.npz"
        np.savez_compressed(output_path, **match_preds)
        logging.info(f"Saved predictions for {match_id} to {output_path}")
