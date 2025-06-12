from pathlib import Path
from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field, root_validator
import yaml
import os


class DataConfig(BaseModel):
    """Data loading configuration."""

    root_dir: Path
    cache_dir: Path
    target_size: tuple[int, int] = (224, 224)  # Default ViT size
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    # If None, uses all data for training
    train_val_split: Optional[float] = None
    val_batch_size: Optional[int] = None
    repeat_train_dataset: Optional[int] = None
    # Dataset selection
    train_dataset_names: List[str] = ["reid_all"]  # Default training datasets
    val_dataset_names: List[str] = ["base"]  # Default validation datasets
    masked_numbers: Optional[List[int]] = None
    # Interpolation method
    interpolation_method: Literal["linear", "cubic"] = "cubic"
    # Training augmentation probabilities
    train_aug_p: float = 0.5
    train_aug_p_scale: float = 0.8


class ModelConfig(BaseModel):
    """Model configuration."""

    model_type: Literal["vit", "unet", "dit"] = "vit"
    architecture: str = "imagenet"
    model_name: str = "vit_base_patch16_224"  # Default ViT model
    pretrained: bool = True
    gradient_clip_val: float = 1.0
    patch_size: int = 16  # Patch size for ViT
    # Optional parameter for OCR training
    finetune_from: Optional[Path] = None
    # Whether to use strict loading when loading checkpoint
    strict_loading: bool = True
    model_kwargs: Optional[dict] = None
    # New parameters for uncertainty modeling
    classifier_type: str = "independent"  # Type of classifier to use
    # Type of position embedding for tied_digit_aware
    embedding_type: str = "additive"
    # Whether to use per-digit bias for tied_digit_aware
    per_digit_bias: bool = True
    # Type of uncertainty modeling
    uncertainty_head: Literal["dirichlet", "softmax"] = "dirichlet"
    size_embedding: bool = False


class LossConfig(BaseModel):
    """Loss function configuration."""

    # Type of loss function
    loss_type: Literal["dirichlet", "softmax"] = "dirichlet"
    # Parameters for Dirichlet loss
    warmup_steps: int = 2500
    reg_weight: float = 0.001
    max_reg_weight: float = 0.05
    # Parameters for Softmax loss
    label_smoothing: float = 0.0


class NoiseFinetuningConfig(BaseModel):
    """Configuration for noise-interpolated finetuning."""

    max_noise: float = 1.0  # Maximum noise level
    noise_warmup: float = 1e-4  # Rate of noise level increase per step


class TrainingConfig(BaseModel):
    """Training configuration."""

    seed: int = 42
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4  # Weight decay for AdamW optimizer
    warmup_steps: int = 100
    final_lr_scale: float = 1e-1
    use_amp: bool = False
    compile_model: bool = False
    logging_steps: int = 10
    # Steps between generating samples/validation
    validation_steps: int = 1000
    save_every_n_epochs: int = 1
    conditional_flow: bool = False
    # Optional parameters for OCR training
    debug: Optional[bool] = None
    debug_samples: Optional[int] = None
    # Optional noise finetuning configuration
    noise_finetuning: Optional[NoiseFinetuningConfig] = None
    save_best_only: bool = True  # Whether to save only the best checkpoint


class LoggingConfig(BaseModel):
    """Logging configuration."""

    base_dir: Path
    # Default for flow matching
    experiment_name: str = Field(default="flow_matching")
    # Parameters for both trainings (with different meanings)
    # Flow: num samples to generate, OCR: num predictions to visualize
    num_samples: int = 16
    num_val_visualizations: int = 16  # Number of validation visualizations

    @property
    def run_dir(self) -> Path:
        """Get the directory for the current training run."""
        # base_dir will be absolute when this is called due to root_validator
        return self.base_dir / self.experiment_name


class Config(BaseModel):
    """Main configuration."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    loss: Optional[LossConfig] = None  # Optional loss configuration

    @root_validator(pre=False, skip_on_failure=True)
    def resolve_paths_in_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure relevant paths are absolute, resolved relative to project root."""
        project_root = Path(os.getenv("OCR_DIR", ".")).resolve()

        data_cfg = values.get("data")
        if data_cfg:
            data_cfg.root_dir = _resolve_path(data_cfg.root_dir, project_root)
            data_cfg.cache_dir = _resolve_path(data_cfg.cache_dir, project_root)

        logging_cfg = values.get("logging")
        if logging_cfg:
            logging_cfg.base_dir = _resolve_path(logging_cfg.base_dir, project_root)

        model_cfg = values.get("model")
        if model_cfg:
            model_cfg.finetune_from = _resolve_path(
                model_cfg.finetune_from, project_root
            )

        return values

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file. Paths resolved by root_validator."""
        project_root = Path(os.getenv("OCR_DIR", ".")).resolve()

        if not yaml_path.is_absolute():
            yaml_path = project_root / yaml_path

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found at {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Pydantic calls root_validator during this instantiation
        return cls(**config_dict)

    def save(self, save_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = self.dict()
        with open(save_path, "w") as f:
            yaml.dump(config_dict, f)

    @property
    def is_flow_matching(self) -> bool:
        """Check if this is a flow matching configuration."""
        return (
            self.data.train_val_split is None
            and self.data.val_batch_size is None
            and self.training.debug is None
        )


# Helper function to resolve paths
def _resolve_path(p: Optional[Path], root: Path) -> Optional[Path]:
    if p is not None and not p.is_absolute():
        return (root / p).resolve()
    # Return resolved absolute path or None
    return p.resolve() if p else None
