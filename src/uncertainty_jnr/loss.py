import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from uncertainty_jnr.model import ModelOutput


class LossWrapper(torch.nn.Module):
    """Wrapper for the loss function to handle the has_prediction mask."""

    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        has_prediction: torch.Tensor,
        step: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of the loss wrapper.

        Args:
            model_output: Output from the model
            targets: Ground truth labels (B,)
            has_prediction: Binary mask indicating if jersey is visible (B,)
            step: Current training step for losses that use warmup

        Returns:
            Loss value or tuple of loss values
        """
        return self.loss_fn(model_output, targets, has_prediction, step)


class Type2DirichletLoss(torch.nn.Module):
    """Implementation of Type II Maximum Likelihood loss with Dirichlet distributions."""

    def __init__(
        self,
        num_classes: int = 100,
        reg_weight: float = 0.01,
        warmup_steps: int = 500,
        max_reg_weight: float = 0.1,
    ):
        """Initialize the loss function.

        Args:
            num_classes: Number of jersey number classes (0-99)
            reg_weight: Initial weight for the regularization term (KL with uniform)
            warmup_steps: Number of steps to linearly increase reg_weight
            max_reg_weight: Maximum regularization weight after warmup
        """
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.warmup_steps = warmup_steps
        self.max_reg_weight = max_reg_weight
        self.uniform_alpha = torch.ones(num_classes)

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        has_prediction: torch.Tensor,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss.

        Args:
            model_output: Output from the model
            targets: Ground truth labels (B,)
            has_prediction: Binary mask indicating if jersey is visible (B,)
            step: Current training step for KL warmup

        Returns:
            Tuple of (total_loss, ml_loss, kl_loss)
        """
        # Use number_logits from model output (excluding absent class)
        logits = model_output.number_logits

        # Get Dirichlet parameters (alpha) by applying exp and adding 1
        alpha = torch.exp(logits) + 1.0  # α = exp(f(x|θ)) + 1

        # Compute S = sum(alpha)
        S = alpha.sum(dim=1)  # (B,)

        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(
            targets, num_classes=self.num_classes
        ).float()  # (B, num_classes)

        # Compute Type II ML loss (equation 3 in the paper)
        # L = sum(y_j * (log(S) - log(alpha_j)))
        ml_loss = torch.sum(
            targets_one_hot * (torch.log(S.unsqueeze(1)) - torch.log(alpha)), dim=1
        )

        # For samples without visible jersey, compute KL with uniform Dirichlet
        # KL(Dir(α₁)||Dir(α₂)) = log(B(α₂)/B(α₁)) + Σ((α₁ᵢ-α₂ᵢ)(ψ(α₁ᵢ)-ψ(Σα₁ⱼ)))
        # where B() is beta function and ψ() is digamma function
        uniform_alpha = self.uniform_alpha.to(alpha.device)

        # Compute digamma terms
        digamma_alpha = torch.digamma(alpha)
        digamma_S = torch.digamma(S.unsqueeze(1))

        # Compute KL divergence
        kl_div = (
            torch.lgamma(S)
            - torch.lgamma(alpha).sum(dim=1)
            - torch.lgamma(torch.tensor(self.num_classes, device=alpha.device))
            + ((alpha - uniform_alpha) * (digamma_alpha - digamma_S)).sum(dim=1)
        )

        # Compute mean losses with masks
        mean_ml_loss = (has_prediction * ml_loss).sum() / (has_prediction.sum() + 1e-6)
        mean_kl_div = ((1 - has_prediction) * kl_div).sum() / (
            (1 - has_prediction).sum() + 1e-6
        )

        # Apply warmup to regularization weight if step is provided
        current_reg_weight = self.reg_weight
        if step is not None:
            progress = min(1.0, step / self.warmup_steps)
            current_reg_weight = self.reg_weight + progress * (
                self.max_reg_weight - self.reg_weight
            )

        total_loss = mean_ml_loss + current_reg_weight * mean_kl_div

        return total_loss, mean_ml_loss, mean_kl_div


class SoftmaxWithUncertaintyLoss(torch.nn.Module):
    """Cross-entropy loss that handles uncertainty by mapping samples without predictions to the absent class."""

    def __init__(self, num_classes: int = 100, label_smoothing: float = 0.0):
        """Initialize the loss function.

        Args:
            num_classes: Number of jersey number classes (0-99)
            label_smoothing: Label smoothing factor for cross-entropy loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        # The total number of classes is num_classes + 1 (for the absent class)
        self.total_classes = num_classes + 1

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
        has_prediction: torch.Tensor,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss.

        Args:
            model_output: Output from the model
            targets: Ground truth labels (B,)
            has_prediction: Binary mask indicating if jersey is visible (B,)
            step: Current training step (not used in this loss)

        Returns:
            Tuple of (total_loss, ce_loss, dummy_loss) for API compatibility
        """
        # Use all_logits from model output (including absent class)
        logits = model_output.all_logits
        batch_size = logits.shape[0]

        # Create modified targets where samples without predictions point to the absent class (index 100)
        modified_targets = targets.clone()
        absent_class_idx = self.num_classes  # Index 100 for the absent class
        modified_targets[has_prediction == 0] = absent_class_idx

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            logits,
            modified_targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Compute mean loss
        mean_ce_loss = ce_loss.mean()

        # Return tuple for API compatibility with DirichletLoss
        # The second and third values are the same since we don't have separate components
        return mean_ce_loss, mean_ce_loss, torch.tensor(0.0, device=logits.device)
