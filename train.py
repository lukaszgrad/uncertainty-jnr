import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import numpy as np
from typing import Optional
import time
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import typing

from uncertainty_jnr.model import TimmOCRModel, ModelOutput
from uncertainty_jnr.loss import Type2DirichletLoss, SoftmaxWithUncertaintyLoss
from uncertainty_jnr.datasets import create_datasets
from uncertainty_jnr.utils import set_seed, save_checkpoint, load_checkpoint
from config import Config

torch.set_float32_matmul_precision("high")


def setup_ddp(rank: int, world_size: int) -> None:
    """Set up distributed data parallel."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355 + 1)  # FIXME
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up distributed data parallel."""
    dist.destroy_process_group()


def get_base_model(model: torch.nn.Module | typing.Callable) -> torch.nn.Module:
    """Get the base nn.Module from a potentially compiled/DDP wrapped model."""
    # Handle compiled model first
    # Check if _orig_mod exists and is an nn.Module
    orig_mod = getattr(model, "_orig_mod", None)
    if isinstance(orig_mod, torch.nn.Module):
        model = orig_mod

    # Then handle DDP wrapper
    if isinstance(model, DDP):
        model = model.module

    # Ensure we are returning a Module after unwrapping
    if not isinstance(model, torch.nn.Module):
        # This should ideally not happen if the input is Module or compiled Module
        raise TypeError(f"Expected nn.Module after unwrapping, but got {type(model)}")
    return model


class RankLogger(logging.LoggerAdapter):
    """Logger adapter that adds rank information to all log records."""

    def __init__(self, logger, rank):
        super().__init__(logger, {})
        self.rank = rank

    def process(self, msg, kwargs):
        return f"[Rank {self.rank}] {msg}", kwargs


def compute_accuracy(
    model_output: ModelOutput,
    targets: torch.Tensor,
    number_mask: Optional[torch.Tensor] = None,
    topk: tuple[int, ...] = (1, 3),
) -> dict:
    """Compute top-k accuracy with optional number masking.

    Args:
        model_output: Model output containing logits and probabilities
        targets: Ground truth labels (B,)
        number_mask: Optional boolean mask for valid numbers (B, num_classes)
        topk: Tuple of k values for top-k accuracy
    """
    # Use number_logits for accuracy computation (excluding absent class)
    logits = model_output.number_logits

    maxk = max(topk)
    batch_size = targets.size(0)

    # Apply number masking if provided
    if number_mask is not None:
        masked_logits = logits.clone()
        masked_logits[~number_mask] = float("-inf")
        _, masked_pred = masked_logits.topk(maxk, 1, True, True)
        masked_pred = masked_pred.t()
        masked_correct = masked_pred.eq(targets.view(1, -1).expand_as(masked_pred))

    # Original unmasked prediction
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        # Unmasked accuracy
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[f"top{k}_acc"] = correct_k.mul_(100.0 / batch_size).item()

        # Masked accuracy if mask provided
        if number_mask is not None:
            masked_correct_k = (
                masked_correct[:k].reshape(-1).float().sum(0, keepdim=True)
            )
            res[f"masked_top{k}_acc"] = masked_correct_k.mul_(100.0 / batch_size).item()

    # Add uncertainty metrics
    uncertainty = model_output.uncertainty
    res["mean_uncertainty"] = uncertainty.mean().item()
    res["max_uncertainty"] = uncertainty.max().item()
    res["min_uncertainty"] = uncertainty.min().item()

    return res


def train_and_evaluate(
    model: torch.nn.Module | typing.Callable,
    train_loader: DataLoader,
    val_loaders: dict[str, DataLoader],
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    config: Config,
    run_dir: Path,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rank: int = 0,
    logger: Optional[RankLogger] = None,
) -> None:
    """Train and evaluate the model."""
    # Create loss function based on configuration
    if config.loss is None or config.loss.loss_type == "dirichlet":
        loss_warmup_steps = config.loss.warmup_steps if config.loss else 2500
        loss_reg_weight = config.loss.reg_weight if config.loss else 0.001
        loss_max_reg_weight = config.loss.max_reg_weight if config.loss else 0.05
        criterion = Type2DirichletLoss(
            warmup_steps=loss_warmup_steps,
            reg_weight=loss_reg_weight,
            max_reg_weight=loss_max_reg_weight,
        )
    # Check config.loss is not None before accessing attributes for softmax
    elif config.loss is not None and config.loss.loss_type == "softmax":
        loss_label_smoothing = config.loss.label_smoothing
        criterion = SoftmaxWithUncertaintyLoss(
            label_smoothing=loss_label_smoothing,
        )
    else:  # Handle case where config.loss is None but loss_type isn't dirichlet (error? fallback?)
        # For now, let's assume a default or raise an error if config.loss is needed but None
        # Defaulting to Dirichlet loss seems safer based on the initial check
        criterion = Type2DirichletLoss()  # Default params
        if rank == 0 and logger:
            logger.warning(
                "Loss configuration missing or invalid, defaulting to DirichletLoss"
            )

    # Use first validation dataset for model selection
    primary_dataset = next(iter(val_loaders.keys()))
    best_val_acc = 0.0
    global_step = 0

    if rank == 0 and logger:
        logger.info(
            f"Using {primary_dataset} as primary validation dataset for model selection"
        )

    for epoch in range(config.training.max_epochs):
        if rank == 0 and logger:
            logger.info(f"\nEpoch {epoch+1}/{config.training.max_epochs}")
        model.train()

        # Set epoch for DistributedSampler
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # Training statistics
        running_loss = 0.0
        running_ml_loss = 0.0
        running_kl_loss = 0.0
        running_acc = 0.0
        running_samples = 0
        epoch_losses = []
        step_start_time = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = batch["gt_number"].to(device)
            has_prediction = batch["has_prediction"].to(device)
            batch_size = images.shape[0]

            size = batch["diagonal"].to(device)

            # Handle noise interpolation if configured
            current_max_noise = 0.0
            t = None
            if config.training.noise_finetuning is not None:
                # Calculate current max noise level based on warmup
                current_max_noise = min(
                    global_step * config.training.noise_finetuning.noise_warmup,
                    config.training.noise_finetuning.max_noise,
                )

                # Sample time uniformly from [0, current_max_noise]
                t = torch.rand(batch_size, 1, device=device) * current_max_noise

                # Generate noisy images
                noise = torch.randn_like(images)
                noisy_images = (1 - t).view(-1, 1, 1, 1) * images + t.view(
                    -1, 1, 1, 1
                ) * noise

                # Use noisy images for training
                images = noisy_images

            optimizer.zero_grad()

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_output = model(images, t=t, size=size)
                    loss, ml_loss, kl_loss = criterion(
                        model_output, targets, has_prediction, step=global_step
                    )

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.model.gradient_clip_val
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                model_output = model(images, t=t, size=size)
                loss, ml_loss, kl_loss = criterion(
                    model_output, targets, has_prediction, step=global_step
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.model.gradient_clip_val
                )
                optimizer.step()

            # Update learning rate
            scheduler.step()

            # Compute accuracy
            metrics = compute_accuracy(model_output, targets)
            acc = metrics["top1_acc"]

            # Update statistics
            running_loss += loss.item() * batch_size
            running_ml_loss += ml_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            running_acc += acc * batch_size
            running_samples += batch_size
            epoch_losses.append(loss.item())
            global_step += 1

            # Log training progress
            if (
                rank == 0
                and step > 0
                and step % config.training.logging_steps == 0
                and logger
            ):
                avg_loss = running_loss / running_samples
                avg_ml_loss = running_ml_loss / running_samples
                avg_kl_loss = running_kl_loss / running_samples
                avg_acc = running_acc / running_samples
                step_time = time.time() - step_start_time
                samples_per_sec = (
                    config.training.logging_steps * batch_size * dist.get_world_size()
                ) / step_time
                logger.info(
                    f"Epoch {epoch+1}, Step {global_step}, "
                    f"Loss: {avg_loss:.4f}, ML Loss: {avg_ml_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, "
                    f"Acc: {avg_acc:.2f}%, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                    f"Samples/sec: {samples_per_sec:.1f}"
                )
                running_loss = 0.0
                running_ml_loss = 0.0
                running_kl_loss = 0.0
                running_acc = 0.0
                running_samples = 0
                step_start_time = time.time()

            # Run validation
            if (
                rank == 0
                and global_step > 0
                and global_step % config.training.validation_steps == 0
            ):
                all_val_metrics = {}
                for dataset_name, val_loader in val_loaders.items():
                    if logger:
                        logger.info(f"\nValidating on {dataset_name} dataset:")

                    # Run validation with clean images
                    val_metrics = validate_step(
                        model=model,
                        val_loader=val_loader,
                        criterion=criterion,
                        device=device,
                        global_step=global_step,
                        run_dir=run_dir / f"{dataset_name}_clean/step_{global_step}",
                        config=config,
                        scaler=scaler,
                        use_noise=False,
                        rank=rank,
                    )
                    all_val_metrics[dataset_name] = val_metrics

                    if logger:
                        logger.info(
                            f"Clean Validation ({dataset_name}, step {global_step}) - "
                            f"Loss: {val_metrics['loss']:.4f} "
                            f"ML Loss: {val_metrics['ml_loss']:.4f} "
                            f"KL Loss: {val_metrics['kl_loss']:.4f} "
                            f"Acc: {val_metrics['top1_acc']:.2f}% "
                            f"Masked Acc: {val_metrics['masked_top1_acc']:.2f}% "
                            f"Top-3 Acc: {val_metrics['top3_acc']:.2f}% "
                            f"Masked Top-3 Acc: {val_metrics['masked_top3_acc']:.2f}%"
                        )

                    # If using noise finetuning, also run validation with noisy images
                    if config.training.noise_finetuning is not None:
                        noisy_val_metrics = validate_step(
                            model=model,
                            val_loader=val_loader,
                            criterion=criterion,
                            device=device,
                            global_step=global_step,
                            run_dir=run_dir
                            / f"{dataset_name}_noisy/step_{global_step}",
                            config=config,
                            scaler=scaler,
                            use_noise=True,
                            current_max_noise=current_max_noise,
                            rank=rank,
                        )

                        if logger:
                            logger.info(
                                f"Noisy Validation ({dataset_name}, step {global_step}) - "
                                f"Loss: {noisy_val_metrics['loss']:.4f} "
                                f"ML Loss: {noisy_val_metrics['ml_loss']:.4f} "
                                f"KL Loss: {noisy_val_metrics['kl_loss']:.4f} "
                                f"Acc: {noisy_val_metrics['top1_acc']:.2f}% "
                                f"Masked Acc: {noisy_val_metrics['masked_top1_acc']:.2f}% "
                                f"Top-3 Acc: {noisy_val_metrics['top3_acc']:.2f}% "
                                f"Masked Top-3 Acc: {noisy_val_metrics['masked_top3_acc']:.2f}%"
                            )

                # Save checkpoint if best on primary validation dataset
                primary_acc = all_val_metrics[primary_dataset]["top1_acc"]
                if primary_acc > best_val_acc:
                    best_val_acc = primary_acc
                    logger.info(
                        f"Saving checkpoint for best validation accuracy: {primary_acc:.2f}%"
                    )
                    save_checkpoint(
                        model=get_base_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_metrics=all_val_metrics,
                        global_step=global_step,
                        run_dir=run_dir,
                        is_best=True,
                        save_best_only=config.training.save_best_only,
                    )

        # Log epoch summary (only on rank 0)
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if rank == 0 and logger:
            logger.info(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")


def validate_step(
    model: torch.nn.Module | typing.Callable,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    global_step: int,
    run_dir: Path,
    config: Config,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_noise: bool = False,
    current_max_noise: Optional[float] = None,
    rank: int = 0,
) -> dict:
    """Run validation step.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    val_loader : DataLoader
        Validation data loader
    criterion : torch.nn.Module
        Loss criterion
    device : torch.device
        Device to run validation on
    global_step : int
        Current training step
    run_dir : Path
        Directory to save validation predictions
    config : Config
        Configuration object
    scaler : Optional[torch.cuda.amp.GradScaler]
        Optional AMP scaler
    use_noise : bool
        Whether to use noise interpolation
    current_max_noise : Optional[float]
        Current maximum noise level (required if use_noise=True)
    rank : int
        Process rank for distributed training
    """
    model.eval()
    val_loss = 0.0
    val_ml_loss = 0.0
    val_kl_loss = 0.0
    val_acc = 0.0
    val_masked_acc = 0.0
    val_samples = 0
    all_metrics = []

    # Create directory for validation predictions (only on rank 0)
    if rank == 0:
        run_dir.mkdir(exist_ok=True, parents=True)

    for idx, batch in enumerate(tqdm(val_loader, desc="Validation", disable=rank != 0)):
        images = batch["image"].to(device)
        targets = batch["gt_number"].to(device)
        has_prediction = batch["has_prediction"].to(device)
        number_mask = batch["number_mask"].to(device)
        batch_size = images.shape[0]

        size = batch["diagonal"].to(device)

        # Add noise if requested
        t = None
        if use_noise and current_max_noise is not None:
            t = torch.rand(batch_size, 1, device=device) * current_max_noise
            noise = torch.randn_like(images)
            images = (1 - t).view(-1, 1, 1, 1) * images + t.view(-1, 1, 1, 1) * noise

        with torch.no_grad():
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_output = model(images, t=t, size=size)
                    loss, ml_loss, kl_loss = criterion(
                        model_output, targets, has_prediction, step=global_step
                    )
            else:
                model_output = model(images, t=t, size=size)
                loss, ml_loss, kl_loss = criterion(
                    model_output, targets, has_prediction, step=global_step
                )

            # Compute metrics
            metrics = compute_accuracy(
                model_output, targets, number_mask=number_mask, topk=(1, 3)
            )
            metrics["loss"] = loss.item()
            metrics["ml_loss"] = ml_loss.item()
            metrics["kl_loss"] = kl_loss.item()
            all_metrics.append(metrics)

            val_loss += loss.item() * batch_size
            val_ml_loss += ml_loss.item() * batch_size
            val_kl_loss += kl_loss.item() * batch_size
            val_acc += metrics["top1_acc"] * batch_size
            val_masked_acc += metrics["masked_top1_acc"] * batch_size
            val_samples += batch_size

            # Save first N predictions as visualizations (only on rank 0)
            if rank == 0 and idx < config.logging.num_val_visualizations:
                # Get Dirichlet parameters
                alpha = (
                    torch.exp(model_output.number_logits[0]) + 1.0
                )  # Take first sample
                S = alpha.sum()

                # Expected probabilities from Dirichlet distribution
                probs = alpha / S
                uncertainty = 100.0 / S.item()

                # Get top predictions
                top_probs, top_preds = probs.topk(3)

                # Create visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                image = images[0].cpu().permute(1, 2, 0).numpy()
                image = (image + 1) * 127.5
                image = image.astype(np.uint8)
                ax.imshow(image)
                ax.axis("off")

                # Add predictions text with uncertainty
                pred_text = (
                    f"GT: {targets[0].item()}\n"
                    f"Uncertainty: {uncertainty:.1%}\n"
                    f"Has Prediction: {bool(has_prediction[0].item())}\n"
                )
                if use_noise:
                    pred_text += f"Noise Level: {t[0].item():.4f}\n"
                pred_text += "-------------------\n"
                for k, (pred, prob) in enumerate(zip(top_preds, top_probs)):
                    pred_text += f"#{k+1}: {pred.item()} ({prob.item():.2%})\n"

                # Color-code title based on uncertainty
                ax.set_title(pred_text, fontsize=12, pad=10)

                # Save plot
                plt.savefig(run_dir / f"pred_{idx}.png", bbox_inches="tight")
                plt.close()

    # Calculate average metrics
    avg_metrics = {
        "loss": val_loss / val_samples,
        "top1_acc": val_acc / val_samples,
        "masked_top1_acc": val_masked_acc / val_samples,
        "ml_loss": val_ml_loss / val_samples,
        "kl_loss": val_kl_loss / val_samples,
    }

    # Add top-3 accuracy
    top3_acc = sum(m["top3_acc"] for m in all_metrics) / len(all_metrics)
    masked_top3_acc = sum(m["masked_top3_acc"] for m in all_metrics) / len(all_metrics)
    avg_metrics["top3_acc"] = top3_acc
    avg_metrics["masked_top3_acc"] = masked_top3_acc

    return avg_metrics


def extract_teams_from_match_id(match_id: str) -> tuple[str, str]:
    """Extract team names from match ID.

    Args:
        match_id: Match ID in format team1-team2-YYYYMMDD

    Returns:
        Tuple of (team1, team2)
    """
    parts = match_id.split("-")
    if len(parts) < 3:
        return "", ""
    return parts[0], parts[1]


def train_worker(
    rank: int,
    world_size: int,
    config: Config,
    run_dir: Path,
    val_datasets: list[str],
) -> None:
    """Worker function for distributed training."""
    # Setup DDP
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Setup logging for this process
    log_file = run_dir / f"log_rank_{rank}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,  # Force reconfiguration of logging
    )

    # Create rank-aware logger
    logger = RankLogger(logging.getLogger(), rank)
    if logger:
        logger.info(f"World size: {world_size}")

    # Set seed for each process to have different shuffling
    set_seed(config.training.seed + rank)

    # Create datasets using the utility function
    train_dataset, val_datasets_dict = create_datasets(
        config=config,
        val_dataset_names=val_datasets,
        train_dataset_names=config.data.train_dataset_names,
        masked_numbers=config.data.masked_numbers,
    )

    # Log dataset information
    if rank == 0 and logger:
        logger.info(f"Using training datasets: {config.data.train_dataset_names}")
        logger.info(f"Training dataset size: {len(train_dataset)} samples")
        for val_name, val_dataset in val_datasets_dict.items():
            logger.info(
                f"Using validation dataset '{val_name}' with {len(val_dataset)} samples"
            )

    # Create samplers and data loaders
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    # Create validation loaders
    val_loaders = {}
    for dataset_name, val_dataset in val_datasets_dict.items():
        val_loaders[dataset_name] = DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size or config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=True,
        )

    # Create model
    model = TimmOCRModel(
        model_name=config.model.model_name,
        pretrained=config.model.pretrained,
        classifier_type=config.model.classifier_type,
        embedding_type=config.model.embedding_type,
        per_digit_bias=config.model.per_digit_bias,
        uncertainty_head=config.model.uncertainty_head,
        size_embedding=config.model.size_embedding,
    )

    # Load checkpoint if provided
    if config.model.finetune_from is not None:
        if rank == 0 and logger:
            logger.info(f"Loading checkpoint from: {config.model.finetune_from}")
        model = load_checkpoint(
            model,
            config.model.finetune_from,
            device,
            strict=config.model.strict_loading,
        )

    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = torch.compile(
        model, mode="default" if world_size == 1 else "reduce-overhead"
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        fused=True,
        weight_decay=1e-3,
    )

    total_steps = len(train_loader) * config.training.max_epochs
    warmup_pct = min(config.training.warmup_steps / total_steps, 0.5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_pct,
        div_factor=25,
        final_div_factor=config.training.final_lr_scale,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None

    try:
        train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loaders=val_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            run_dir=run_dir,
            scaler=scaler,
            rank=rank,
            logger=logger,
        )

        if rank == 0 and logger:
            logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        if rank == 0 and logger:
            logger.info("Training interrupted by user")
            save_checkpoint(
                model=get_base_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                val_metrics={"interrupted": True},
                global_step=-1,
                run_dir=run_dir,
                is_best=False,
                save_best_only=config.training.save_best_only,
            )
            if logger:
                logger.info("Saved interrupt checkpoint")

    except Exception as e:
        if rank == 0 and logger:
            logger.exception("Training failed with error:")
        raise e

    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(
        description="Train jersey number recognition model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--override-dir",
        action="store_true",
        help="Override run directory if it exists",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix to append to run directory name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed value from config",
    )
    parser.add_argument(
        "--masked-numbers",
        type=int,
        nargs="+",
        default=None,
        help="List of jersey numbers to mask during training",
    )
    args = parser.parse_args()

    # Load config - from_yaml now handles path resolution internally
    # The config path itself is resolved within from_yaml based on OCR_DIR
    config = Config.from_yaml(args.config)

    # Override seed if specified
    if args.seed is not None:
        config.training.seed = args.seed

    # Add masked numbers to config if specified
    if args.masked_numbers is not None:
        if not hasattr(config.data, "masked_numbers"):
            config.data.masked_numbers = args.masked_numbers
        else:
            config.data.masked_numbers = args.masked_numbers

    # Create run directory (uses the now absolute config.logging.base_dir)
    run_dir = config.logging.run_dir

    # Apply suffix if provided
    if args.suffix:
        run_dir = run_dir.with_name(f"{run_dir.name}_{args.suffix}")

    # If seed is provided and no suffix, add seed as suffix
    if args.seed is not None and not args.suffix:
        run_dir = run_dir.with_name(f"{run_dir.name}_seed{args.seed}")

    # Check if directory exists and handle based on override flag
    if run_dir.exists() and not args.override_dir:
        raise ValueError(
            f"Run directory {run_dir} already exists. Use --override-dir to override."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(run_dir / "config.yaml")

    # Get world size from CUDA_VISIBLE_DEVICES or number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(world_size, config, run_dir, config.data.val_dataset_names),
            nprocs=world_size,
            join=True,
        )
    else:
        # Fall back to single GPU training
        train_worker(0, 1, config, run_dir, config.data.val_dataset_names)


if __name__ == "__main__":
    main()
