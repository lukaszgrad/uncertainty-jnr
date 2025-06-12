import albumentations as A
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal


# Minimum size for the shorter side after scaling
MIN_SCALE_SIZE = 32


INTERPOLATION_MAP = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


@dataclass
class ColorShiftConfig:
    """Configuration for color shift augmentation."""

    green_h_range: tuple[int, int] = (35, 85)  # Default HSV green range
    color_margin: int = 8  # Margin around dominant H value
    noise_range: tuple[int, int] = (-50, 50)  # Range for random H noise


class DominantColorShift(A.ImageOnlyTransform):
    """Custom augmentation that shifts the dominant color in the image.

    Finds the dominant color (excluding green pitch) and adds random noise
    to pixels of similar hue.
    """

    def __init__(
        self, config: ColorShiftConfig, always_apply: bool = False, p: float = 0.5
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.config = config

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Create mask for non-green pixels
        green_lower = np.array(self.config.green_h_range[0], dtype=np.uint8)
        green_upper = np.array(self.config.green_h_range[1], dtype=np.uint8)
        green_mask = ~cv2.inRange(h, green_lower, green_upper)

        # Find dominant hue using histogram
        hist = cv2.calcHist([h], [0], green_mask, [180], [0, 180])
        dominant_h = np.argmax(hist)

        # Create mask for pixels with similar hue to dominant color
        color_lower = np.array(
            max(0, dominant_h - self.config.color_margin), dtype=np.uint8
        )
        color_upper = np.array(
            min(179, dominant_h + self.config.color_margin), dtype=np.uint8
        )
        color_mask = cv2.inRange(h, color_lower, color_upper)

        # Generate random noise for selected pixels
        noise = np.random.randint(
            self.config.noise_range[0],
            self.config.noise_range[1],
            (1,),
            dtype=np.int16,
        )

        # Apply noise only to selected pixels
        h = h.astype(np.int16)
        h[(color_mask > 0) & (green_mask > 0)] += noise
        h = np.clip(h, 0, 179).astype(np.uint8)

        # Merge channels back
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def get_transform_init_args_names(self) -> tuple:
        """Get the names of the arguments used in __init__."""
        return ("config",)


class JerseyCrop(A.ImageOnlyTransform):
    """Custom augmentation that crops the image to focus on jersey area.

    Removes bottom 1/3 (legs) and top 1/6 (head) of the image to focus on torso area
    where jersey number is typically located.
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        height = img.shape[0]

        # Calculate crop boundaries
        top_crop = height // 6  # Remove top 1/6th
        bottom_crop = height - (height // 3)  # Remove bottom 1/3rd

        # Crop the image
        return img[top_crop:bottom_crop, :]

    def get_transform_init_args_names(self) -> tuple:
        """Get the names of the arguments used in __init__."""
        return ()


class RandomScaling(A.ImageOnlyTransform):
    """Custom augmentation that randomly scales the image.

    Ensures the shorter side of the image is not less than MIN_SCALE_SIZE pixels.
    If the original image is already smaller than MIN_SCALE_SIZE, no scaling is performed.
    """

    def __init__(
        self,
        interpolation_method: Literal["linear", "cubic"] = "cubic",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.interpolation = INTERPOLATION_MAP[interpolation_method]

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        height, width = img.shape[:2]
        shorter_side = min(height, width)

        # If the image is already smaller than MIN_SCALE_SIZE, return it as is
        if shorter_side < MIN_SCALE_SIZE:
            return img

        # Calculate the current scale factor (1.0 means no scaling)
        current_scale = MIN_SCALE_SIZE / shorter_side

        # Generate a random scale factor between current_scale and 1.0
        # This ensures the shorter side is not less than MIN_SCALE_SIZE
        scale_factor = np.random.uniform(current_scale, 1.0)

        # Calculate new dimensions while preserving aspect ratio
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=self.interpolation
        )

        return resized_img

    def get_transform_init_args_names(self) -> tuple:
        """Get the names of the arguments used in __init__."""
        return ("interpolation_method",)


def get_train_transforms(
    target_size: tuple[int, int],
    interpolation_method: Literal["linear", "cubic"] = "cubic",
    center_crop: bool = True,
    p: float = 0.5,
    p_scale: float = 0.0,
) -> A.Compose:
    """Get default training augmentations.

    Args:
        target_size: Target image size (height, width)
        interpolation_method: Interpolation method ('linear' or 'cubic')
        center_crop: Whether to apply jersey center cropping
        p: Probability of applying each augmentation
        p_scale: Probability of applying random scaling (0.0 means no scaling)

    Returns:
        Composed augmentation pipeline
    """
    transforms = []
    interpolation = INTERPOLATION_MAP[interpolation_method]

    # Add random scaling if p_scale > 0 (Original Position)
    if p_scale > 0:
        transforms.append(
            RandomScaling(p=p_scale, interpolation_method=interpolation_method)
        )

    # Capture size after potential scaling (Original Position)
    transforms.append(SizeCapture(always_apply=True))

    # Add jersey crop if requested (Original Position)
    if center_crop:
        transforms.append(JerseyCrop(always_apply=True))

    # Add other augmentations (Original Position)
    transforms.extend(
        [
            DominantColorShift(config=ColorShiftConfig(), p=p),
            # A.HorizontalFlip(p=p),  # TODO: we can't flip the jersey
            A.Rotate(limit=15, p=p),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0.5, p=1.0),
                ],
                p=p,
            ),
            # Final resize to target size (Moved back to original position)
            A.Resize(
                height=target_size[0],
                width=target_size[1],
                always_apply=True,
                interpolation=interpolation,
            ),
        ]
    )

    return A.Compose(transforms)


def get_val_transforms(
    target_size: tuple[int, int],
    interpolation_method: Literal["linear", "cubic"] = "cubic",
    center_crop: bool = True,
) -> A.Compose:
    """Get validation transforms (resize and optional crop).

    Args:
        target_size: Target image size (height, width)
        interpolation_method: Interpolation method ('linear' or 'cubic')
        center_crop: Whether to apply jersey center cropping

    Returns:
        Composed transform pipeline
    """
    transforms = []
    interpolation = INTERPOLATION_MAP[interpolation_method]

    transforms.append(SizeCapture(always_apply=True))

    # Add jersey crop if requested
    if center_crop:
        transforms.append(JerseyCrop(always_apply=True))

    # Add resize
    transforms.append(
        A.Resize(
            height=target_size[0],
            width=target_size[1],
            always_apply=True,
            interpolation=interpolation,
        )
    )

    return A.Compose(transforms)


class SizeCapture(A.ImageOnlyTransform):
    """Transform that captures image dimensions after all transformations.

    This transform doesn't modify the image but records its dimensions in the
    returned dict when applied as the last transform in a pipeline.
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Return the image unchanged - just capture size
        params["width"] = img.shape[1]
        params["height"] = img.shape[0]
        return img

    def get_transform_init_args_names(self) -> tuple:
        """Get the names of the arguments used in __init__."""
        return ()
