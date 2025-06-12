import torch
import timm
from typing import Dict, Any, Optional, Literal
from timm.models.vision_transformer import VisionTransformer
from timm.models.eva import Eva
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Structured output from the OCR model.

    This class provides a consistent interface for different uncertainty modeling approaches.
    """

    all_logits: (
        torch.Tensor
    )  # Raw logits from the model (B, 101) - includes absent class
    number_logits: torch.Tensor  # Logits for jersey numbers only (B, 100)
    number_probs: torch.Tensor  # Probabilities for jersey numbers (B, 100)
    uncertainty: torch.Tensor  # Uncertainty score (B,) - higher means more uncertain


class JerseyClassifier(ABC, torch.nn.Module):
    """Abstract base class for jersey number classifiers."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for all 100 numbers plus absent class (101 total)."""
        pass


class IndependentClassifier(JerseyClassifier):
    """Classical approach treating each number as independent class, with absent class."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)
        # Add one more output for the "absent" class (101 total)
        self.classifier = torch.nn.Linear(embed_dim, 101)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DigitAwareClassifier(JerseyClassifier):
    """Digit-aware classifier that reconstructs numbers from digit predictions, with absent class."""

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim)

        # Three separate digit classifiers
        self.single_digit = torch.nn.Linear(embed_dim, 10)  # 0-9
        self.tens_digit = torch.nn.Linear(embed_dim, 10)  # 0-9 for tens place
        self.ones_digit = torch.nn.Linear(embed_dim, 10)  # 0-9 for ones place

        # Separate linear layer for the "absent" class
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)

        # Register buffer for indices to avoid recreation
        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get digit logits
        single_logits = self.single_digit(x)  # (B, 10)
        tens_logits = self.tens_digit(x)  # (B, 10)
        ones_logits = self.ones_digit(x)  # (B, 10)

        # Get absent logit
        absent_logit = self.absent_classifier(x)  # (B, 1)

        batch_size = x.shape[0]

        # Initialize output logits for numbers (0-99)
        number_logits = torch.empty(batch_size, 100, device=x.device)

        # Fill single digits (0-9)
        number_logits[:, :10] = single_logits

        # Fill two-digit numbers (10-99) using vectorized operations
        # Shape: (B, 90)
        two_digit_logits = tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        number_logits[:, 10:] = two_digit_logits

        # Concatenate with absent logit to get all logits (0-99 + absent)
        all_logits = torch.cat([number_logits, absent_logit], dim=1)  # (B, 101)

        return all_logits


class TiedDigitAwareClassifier(JerseyClassifier):
    """Digit-aware classifier with shared weights and position embeddings, with absent class."""

    def __init__(
        self,
        embed_dim: int,
        embedding_type: str = "additive",
        per_digit_bias: bool = True,
    ):
        """Initialize the classifier.

        Args:
            embed_dim: Dimension of input embeddings
            embedding_type: Type of position embedding ('additive' or 'multiplicative')
            per_digit_bias: Whether to use per-digit bias
        """
        super().__init__(embed_dim)

        if embedding_type not in ["additive", "multiplicative"]:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        self.embedding_type = embedding_type
        self.per_digit_bias = per_digit_bias
        # Single digit classifier shared across positions (without bias)
        self.digit_classifier = torch.nn.Linear(embed_dim, 10, bias=False)

        # Separate linear layer for the "absent" class
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)

        # Learnable position-specific biases
        if per_digit_bias:
            self.position_biases = torch.nn.Parameter(torch.zeros(3, 10))
        else:
            self.position_biases = torch.nn.Parameter(torch.zeros(3, 1))

        # Learnable position embeddings - initialize close to 1 for multiplicative
        init_value = (
            torch.randn(3, embed_dim)
            if embedding_type == "additive"
            else torch.ones(3, embed_dim)
        )
        self.position_embeddings = torch.nn.Parameter(init_value)

        # Register buffer for indices
        tens_idx = torch.div(torch.arange(10, 100), 10, rounding_mode="floor")
        ones_idx = torch.remainder(torch.arange(10, 100), 10)
        self.register_buffer("tens_idx", tens_idx)
        self.register_buffer("ones_idx", ones_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Apply position embeddings based on type
        if self.embedding_type == "additive":
            positioned_features = x.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:  # multiplicative
            positioned_features = x.unsqueeze(1) * self.position_embeddings.unsqueeze(0)

        digit_logits = self.digit_classifier(positioned_features)
        digit_logits = digit_logits + self.position_biases.unsqueeze(0)

        single_logits = digit_logits[:, 0]
        tens_logits = digit_logits[:, 1]
        ones_logits = digit_logits[:, 2]

        # Get absent logit
        absent_logit = self.absent_classifier(x)  # (B, 1)

        # Initialize output logits for numbers (0-99)
        number_logits = torch.empty(batch_size, 100, device=x.device)
        number_logits[:, :10] = single_logits
        two_digit_logits = tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        number_logits[:, 10:] = two_digit_logits

        # Concatenate with absent logit to get all logits (0-99 + absent)
        all_logits = torch.cat([number_logits, absent_logit], dim=1)  # (B, 101)

        return all_logits


class TimmOCRModel(torch.nn.Module):
    """OCR model based on pretrained timm Vision Transformer."""

    def __init__(
        self,
        model_name: str,
        classifier_type: str = "independent",
        embedding_type: str = "additive",
        per_digit_bias: bool = True,
        uncertainty_head: Literal["dirichlet", "softmax"] = "dirichlet",
        pretrained: bool = True,
        size_embedding: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the OCR model.

        Args:
            model_name: Name of the timm model to use (must be a Vision Transformer)
            classifier_type: Type of classifier to use ('independent', 'digit_aware', or 'tied_digit_aware')
            embedding_type: Type of position embedding for tied_digit_aware ('additive' or 'multiplicative')
            per_digit_bias: Whether to use per-digit bias for tied_digit_aware
            uncertainty_head: Type of uncertainty modeling ('dirichlet' or 'softmax')
            pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments to pass to timm.create_model
        """
        super().__init__()

        # Create the backbone model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, **kwargs)

        # Verify that it's a Vision Transformer or Eva
        if not isinstance(self.backbone, (VisionTransformer, Eva)):
            raise ValueError(
                f"Model {model_name} is not a Vision Transformer. "
                f"Got {type(self.backbone)} instead."
            )

        # Get embedding dimension from the model
        embed_dim = self.backbone.embed_dim

        # Time embedding layer with small initialization
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
        )
        self.size_embed = (
            torch.nn.Sequential(
                torch.nn.Linear(1, embed_dim),
            )
            if size_embedding
            else None
        )
        # Initialize embedding with small values
        with torch.no_grad():
            self.time_embed[0].weight.data.uniform_(-0.001, 0.001)
            self.time_embed[0].bias.data.zero_()

            if self.size_embed is not None:
                self.size_embed[0].weight.data.uniform_(-0.003, 0.003)
                self.size_embed[0].bias.data.zero_()

        # Create classifier based on type
        if classifier_type == "independent":
            self.classifier = IndependentClassifier(embed_dim)
        elif classifier_type == "digit_aware":
            self.classifier = DigitAwareClassifier(embed_dim)
        elif classifier_type == "tied_digit_aware":
            self.classifier = TiedDigitAwareClassifier(
                embed_dim,
                embedding_type=embedding_type,
                per_digit_bias=per_digit_bias,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Store uncertainty head type
        if uncertainty_head not in ["dirichlet", "softmax"]:
            raise ValueError(f"Unknown uncertainty head: {uncertainty_head}")
        self.uncertainty_head = uncertainty_head

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input images
        t : Optional[torch.Tensor]
            Optional time conditioning of shape (B, 1)
        size : Optional[torch.Tensor]
            Optional size conditioning of shape (B, 2)

        Returns
        -------
        ModelOutput
            Structured output containing logits, probabilities, and uncertainty
        """
        # Get features from backbone
        features = self.forward_features(x, t, size)

        # Apply classification head to get all logits (including absent class)
        all_logits = self.classifier(features)  # (B, 101)

        # Extract number logits (excluding absent class)
        number_logits = all_logits[:, :100]  # (B, 100)

        if self.uncertainty_head == "dirichlet":
            # Convert logits to Dirichlet parameters and probabilities
            alpha = torch.exp(number_logits) + 1.0  # α = exp(f(x|θ)) + 1
            S = alpha.sum(dim=1, keepdim=True)
            number_probs = alpha / S

            # Compute uncertainty as inverse of concentration parameter
            # Higher value means more uncertain
            uncertainty = 100.0 / S.squeeze()
        else:  # softmax
            # Apply softmax to all logits (including absent class)
            all_probs = torch.nn.functional.softmax(all_logits, dim=1)

            # Extract number probabilities and absent probability
            number_probs = all_probs[:, :100]  # (B, 100)
            absent_prob = all_probs[:, 100]  # (B,)

            # Use absent probability as uncertainty
            uncertainty = absent_prob

        return ModelOutput(
            all_logits=all_logits,
            number_logits=number_logits,
            number_probs=number_probs,
            uncertainty=uncertainty,
        )

    def forward_features(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the embedding vector before classification.

        Parameters
        ----------
        x : torch.Tensor
            Input images
        t : Optional[torch.Tensor]
            Optional time conditioning of shape (B, 1)
        size : Optional[torch.Tensor]
            Optional size conditioning of shape (B, 2)
        """
        # Patch embedding
        x = self.backbone.patch_embed(x)

        # Add position embeddings
        x = self.backbone._pos_embed(x)

        # Add time embeddings if provided
        if t is not None:
            t_emb = self.time_embed(t)
            x = x + t_emb.unsqueeze(1)  # Add to all positions

        if size is not None and self.size_embed is not None:
            size_emb = self.size_embed(size)
            x[:, 1:] = x[:, 1:] + size_emb.unsqueeze(
                1
            )  # Add to all positions except CLS token

        # Process through rest of backbone
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        # Get pre-logits features
        x = x[:, 0]  # Use CLS token
        return x

    def compile(self):
        """Compile the model."""
        self.backbone = torch.compile(self.backbone)
