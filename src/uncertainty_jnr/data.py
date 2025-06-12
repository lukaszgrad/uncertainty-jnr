from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import logging
from dataclasses import dataclass
import hashlib
import pickle


@dataclass
class MatchData:
    """Container for match-specific data files."""

    match_id: str
    half: str
    track_file: Path
    jersey_gt_file: Optional[Path]  # Now optional
    jersey_pred_file: Optional[Path]  # Now optional
    detections_dir: Path
    possible_numbers: set[int]
    has_gt: bool = True  # Flag to indicate if GT data is available
    has_pred: bool = True  # Flag to indicate if prediction data is available


class JerseyNumberDataset(Dataset):
    """Dataset for jersey number recognition."""

    def __init__(
        self,
        root_dir: Path | str,
        match_ids: list[str] | None = None,
        target_size: tuple[int, int] = (128, 64),
        transform: Optional[Callable] = None,
        cache_dir: Optional[Path] = None,
        filter_invalid: bool = False,
        limit_samples: Optional[int] = None,
        inference_mode: bool = False,
        masked_numbers: list[int] | None = None,
    ):
        """Initialize dataset.

        Args:
            root_dir: Root directory containing match data
            match_ids: Optional list of match IDs to use. If None, use all matches
            target_size: Target size for images (height, width)
            transform: Optional transform to apply to images
            cache_dir: Optional cache directory to save dataset
            filter_invalid: Whether to filter out invalid jersey numbers
            limit_samples: Optional limit on number of samples (randomly selected)
            inference_mode: If True, don't require jersey GT or prediction files
            masked_numbers: Optional list of jersey numbers to mask during training
        """
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.filter_invalid = filter_invalid
        self.limit_samples = limit_samples
        self.inference_mode = inference_mode  # Store the inference mode flag
        self.masked_numbers = set(masked_numbers) if masked_numbers else set()

        # Get all available match IDs
        all_match_ids = [p.name for p in self.root_dir.glob("*") if p.is_dir()]

        # Filter matches if specified
        self.match_ids = sorted(match_ids if match_ids is not None else all_match_ids)
        self.logger.info(f"Using {len(self.match_ids)} matches: {self.match_ids}")

        if self.masked_numbers:
            self.logger.info(f"Masking jersey numbers: {sorted(self.masked_numbers)}")

        # Try to load from cache first
        cache_hit = False
        if self.cache_dir is not None:
            cache_hash = self._compute_cache_hash()
            cache_path = self.cache_dir / cache_hash
            if cache_path.exists():
                self.logger.info(f"Loading dataset from cache: {cache_path}")
                try:
                    with open(cache_path / "cache.pkl", "rb") as f:
                        cache_data = pickle.load(f)
                    self.match_data = cache_data["match_data"]
                    self.detection_df = cache_data["detection_df"]
                    cache_hit = True
                    self.logger.info("Successfully loaded from cache")
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")

        # If not in cache, compute and save
        if not cache_hit:
            self.logger.info("Computing dataset from scratch")
            # Find and validate match data
            self.match_data = self._find_valid_matches()
            # Create detection index and load all required data
            self.detection_df = self._create_detection_index()

            # Save to cache if enabled
            if self.cache_dir is not None:
                self.logger.info(f"Saving dataset to cache: {cache_path}")
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "match_data": self.match_data,
                    "detection_df": self.detection_df,
                }
                with open(cache_path / "cache.pkl", "wb") as f:
                    pickle.dump(cache_data, f)

        # Apply masking if needed (after loading from cache or creating from scratch)
        if self.masked_numbers:
            self._apply_number_masking()

        # Apply sample limit if specified (after caching to keep cache consistent)
        if (
            self.limit_samples is not None
            and len(self.detection_df) > self.limit_samples
        ):
            self.logger.info(f"Limiting dataset to {self.limit_samples} random samples")
            self.detection_df = self.detection_df.sample(
                n=self.limit_samples, random_state=42
            )

    def _find_valid_matches(self) -> dict[str, MatchData]:
        """Find all valid match directories by looking for track.csv files."""
        match_data = {}

        # Only look for matches in self.match_ids
        for match_id in self.match_ids:
            match_path = self.root_dir / match_id

            # Find all track.csv files for this match
            csv_files = list(match_path.glob("**/detection/track.csv"))
            feather_files = list(match_path.glob("**/detection/track.feather"))
            if not csv_files and not feather_files:
                self.logger.warning(
                    f"No track.csv or track.feather files found for {match_id}"
                )
                continue
            if len(csv_files) > len(feather_files):
                suffix = ".csv"
                files = csv_files
            else:
                suffix = ".feather"
                files = feather_files

            for track_file in files:
                match_dir = track_file.parent.parent

                # Get half from path
                half = match_dir.name
                match_key = f"{match_id}/{half}"

                # Check for required files
                jersey_gt = (
                    match_dir / "track" / "jersey_number_aggregated-annotated.csv"
                )
                jersey_pred = match_dir / "detection" / f"jersey_number{suffix}"
                detections_dir = match_dir / "input_detections"

                # In inference mode, jersey files are optional
                has_gt = jersey_gt.exists()
                has_pred = jersey_pred.exists()

                # Check if we have the minimum required files
                if not detections_dir.exists():
                    self.logger.warning(
                        f"Skipping match data without detections directory: {match_dir}"
                    )
                    continue

                # In non-inference mode, we require both jersey files
                if not self.inference_mode and not (has_gt and has_pred):
                    self.logger.warning(
                        f"Skipping incomplete match data in: {match_dir}"
                    )
                    continue

                match_data[match_key] = MatchData(
                    match_id=match_id,
                    half=half,
                    track_file=track_file,
                    jersey_gt_file=jersey_gt if has_gt else None,
                    jersey_pred_file=jersey_pred if has_pred else None,
                    detections_dir=detections_dir,
                    possible_numbers=set(),
                    has_gt=has_gt,
                    has_pred=has_pred,
                )

        self.logger.info(
            f"Found {len(match_data)} valid match halves for {len(self.match_ids)} matches"
        )
        return match_data

    def _create_detection_index(self) -> pd.DataFrame:
        """Create index of all valid detections with their GT and predictions."""
        all_data = []
        match_jerseys = {}

        for match_key, data in self.match_data.items():
            # Get all detection images
            detection_files = list(data.detections_dir.glob("*.png"))
            if not detection_files:
                continue

            # Extract file_name and detection_id from image paths
            detections = pd.DataFrame(
                [
                    {
                        "match_key": match_key,
                        "file_name": int(p.stem.split("_")[0]),
                        "detection_id": int(float(p.stem.split("_")[1])),
                    }
                    for p in detection_files
                ]
            )

            # Load and merge track data
            try:
                track_df = (
                    pd.read_csv(data.track_file)
                    if data.track_file.suffix == ".csv"
                    else pd.read_feather(data.track_file)
                )
            except Exception as e:
                self.logger.warning(f"Failed to load track file for {match_key}: {e}")
                continue

            detections = detections.merge(
                track_df, on=["file_name", "detection_id"], how="left"
            )

            # Load GT jersey numbers if available
            if data.has_gt and data.jersey_gt_file is not None:
                try:
                    gt_df = pd.read_csv(data.jersey_gt_file)

                    if gt_df.jersey_number.dtype == "object":
                        self.logger.warning(
                            f"Jersey number is object type in {match_key}, skipping GT"
                        )
                        data.has_gt = False
                    elif gt_df.jersey_number_score.dtype == "object":
                        self.logger.warning(
                            f"Jersey number score is object type in {match_key}, skipping GT"
                        )
                        data.has_gt = False
                    else:
                        # Validate jersey numbers before grouping
                        invalid_numbers = gt_df[~gt_df["jersey_number"].between(0, 99)]
                        if not invalid_numbers.empty and self.filter_invalid:
                            self.logger.warning(
                                f"Found {len(invalid_numbers)} invalid jersey numbers in {match_key}: "
                                f"{invalid_numbers['jersey_number'].unique().tolist()}"
                            )
                            # Filter out invalid numbers
                            gt_df = gt_df[gt_df["jersey_number"].between(0, 99)]

                        valid_numbers = set(gt_df["jersey_number"].unique()) & set(
                            range(100)
                        )
                        self.match_data[match_key].possible_numbers = valid_numbers

                        # Filter only observed tracks and get the jersey number with highest score
                        gt_numbers = (
                            gt_df.sort_values("jersey_number_score", ascending=False)
                            .groupby("track_id")
                            .first()
                            .reset_index()[["track_id", "jersey_number"]]
                        )

                        detections = detections.merge(
                            gt_numbers, on="track_id", how="left", suffixes=("", "_gt")
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load jersey GT file for {match_key}: {e}"
                    )
                    data.has_gt = False

            # If GT is not available, add dummy GT column
            if not data.has_gt:
                detections["jersey_number_gt"] = -1
                self.match_data[match_key].possible_numbers = set(
                    range(100)
                )  # All numbers are possible

            # Load and merge predictions if available
            if data.has_pred and data.jersey_pred_file is not None:
                try:
                    pred_df = (
                        pd.read_csv(data.jersey_pred_file)
                        if data.jersey_pred_file.suffix == ".csv"
                        else pd.read_feather(data.jersey_pred_file)
                    )

                    if "score" not in pred_df.columns:
                        self.logger.warning(
                            f"Match {match_key} predictions are missing the score column, skipping predictions"
                        )
                        data.has_pred = False
                    else:
                        pred_df = pred_df.sort_values(
                            "score", ascending=False
                        ).drop_duplicates(["file_name", "detection_id"])
                        detections = detections.merge(
                            pred_df[
                                ["file_name", "detection_id", "jersey_number", "score"]
                            ],
                            on=["file_name", "detection_id"],
                            how="left",
                            suffixes=("_gt", "_pred"),
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load jersey prediction file for {match_key}: {e}"
                    )
                    data.has_pred = False

            # If predictions are not available, add dummy prediction columns
            if not data.has_pred:
                detections["jersey_number_pred"] = -1
                detections["score"] = 0.0

            all_data.append(detections)

        # Combine all data
        detection_df = pd.concat(all_data, ignore_index=True)

        # Final validation of GT numbers if not in inference mode
        if not self.inference_mode:
            invalid_samples = detection_df[
                ~detection_df["jersey_number_gt"].between(0, 99)
            ]
            if not invalid_samples.empty and self.filter_invalid:
                self.logger.warning(
                    f"Filtering out {len(invalid_samples)} samples with invalid GT numbers"
                )
                detection_df = detection_df[
                    detection_df["jersey_number_gt"].between(0, 99)
                ]

            # Filter valid samples (those without GT jersey number)
            detection_df = detection_df.dropna(subset=["jersey_number_gt"])

        self.logger.info(f"Created index with {len(detection_df)} valid detections")
        return detection_df

    def _get_possible_img_paths(
        self, match_key: str, file_name: int, detection_id: int
    ) -> tuple[Path, Path, Path, Path]:
        img_path_padded_int = (
            self.match_data[match_key].detections_dir
            / f"{file_name:06d}_{detection_id}.png"
        )
        img_path_raw_int = (
            self.match_data[match_key].detections_dir
            / f"{file_name}_{detection_id}.png"
        )
        img_path_padded_float = (
            self.match_data[match_key].detections_dir
            / f"{file_name:06d}_{detection_id:.1f}.png"
        )
        img_path_raw_float = (
            self.match_data[match_key].detections_dir
            / f"{file_name}_{detection_id:.1f}.png"
        )
        return (
            img_path_padded_int,
            img_path_raw_int,
            img_path_padded_float,
            img_path_raw_float,
        )

    def _load_image(
        self, match_key: str, file_name: int, detection_id: int
    ) -> tuple[torch.Tensor, int, int, Path]:
        """Load and preprocess detection image.

        Args:
            match_key: Match key in format match_id/half
            file_name: Frame number
            detection_id: Detection ID within the frame

        Returns:
            Tuple of (image tensor, width, height, image path)
        """
        img_paths = self._get_possible_img_paths(match_key, file_name, detection_id)
        for img_path in img_paths:
            if img_path.exists():
                break

        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning(f"Failed to load image: {img_path}")
            return (
                torch.zeros(
                    (3, self.target_size[0], self.target_size[1]), dtype=torch.float32
                ),
                64,
                128,
                img_path,
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = img.shape[:2]

        # Apply albumentations transforms if provided
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

            # Use the width and height captured by SizeCapture if available
            # otherwise fall back to original dimensions
            width = transformed.get("width", orig_width)
            height = transformed.get("height", orig_height)
        else:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            width = orig_width
            height = orig_height

        # Convert to tensor and normalize
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return (img.float() / 127.5) - 1.0, width, height, img_path

    def __len__(self) -> int:
        return len(self.detection_df)

    @property
    def possible_numbers(self) -> list[int]:
        """Get all valid jersey numbers from all matches in the dataset.

        Returns:
            List of all valid jersey numbers (0-99) that appear in the dataset.
        """
        all_numbers = set()
        for match_data in self.match_data.values():
            all_numbers.update(match_data.possible_numbers)
        return sorted(list(all_numbers))

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset."""
        row = self.detection_df.iloc[idx]
        match_key = row.match_key
        match_data = self.match_data[match_key]

        # Create one-hot encoded mask for possible numbers
        number_mask = torch.zeros(100, dtype=torch.bool)
        number_mask[list(match_data.possible_numbers)] = True

        image, width, height, image_path = self._load_image(
            row.match_key, row.file_name, row.detection_id
        )
        diagonal = torch.Tensor([np.sqrt(width**2 + height**2)])
        diagonal = diagonal.clamp(
            min=32, max=128
        )  # TODO: find appriopriate limits based on training dataset

        return {
            "image": image,
            "gt_number": (
                int(row.jersey_number_gt) if pd.notna(row.jersey_number_gt) else -1
            ),
            "pred_number": (
                int(row.jersey_number_pred) if pd.notna(row.jersey_number_pred) else -1
            ),
            "pred_score": float(row.score) if pd.notna(row.score) else 0.0,
            "has_prediction": int(pd.notna(row.jersey_number_pred)),
            "has_gt": int(match_data.has_gt),  # New field indicating if GT is available
            "number_mask": number_mask,
            "match_key": match_key,
            "file_name": int(row.file_name),
            "detection_id": int(row.detection_id),
            "track_id": int(row.track_id) if pd.notna(row.track_id) else -1,
            "image_path": str(
                image_path.relative_to(self.root_dir)
            ),  # Save relative path
            "width": width,
            "height": height,
            "diagonal": diagonal,
        }

    def _compute_cache_hash(self) -> str:
        """Compute a hash for the dataset to use as cache key.

        Note: We intentionally don't include limit_samples or masked_numbers in the hash
        to keep the cache consistent. Masking is applied after loading from cache.
        """
        data = [
            self.root_dir,
            self.match_ids,
            self.filter_invalid,
        ]
        if self.inference_mode:
            data.append(self.inference_mode)
        return hashlib.sha256(pickle.dumps(data)).hexdigest()

    def _apply_number_masking(self) -> None:
        """Apply masking to jersey numbers in the dataset.

        This is done after loading from cache or creating from scratch.
        Instead of setting values to NaN, we filter out rows with masked numbers.
        """
        if not self.masked_numbers:
            return

        total_masked_samples = 0
        match_jerseys = {}
        original_size = len(self.detection_df)

        # Process each match separately
        for match_key, data in self.match_data.items():
            # Skip if no GT data
            if not data.has_gt:
                continue

            # Get match-specific rows
            match_rows = self.detection_df[self.detection_df["match_key"] == match_key]

            # Count samples before masking
            before_count = match_rows.dropna(subset=["jersey_number_gt"]).shape[0]

            # Identify entries with jersey numbers in the masked list
            mask_condition = match_rows["jersey_number_gt"].isin(self.masked_numbers)
            masked_count = mask_condition.sum()

            if masked_count > 0:
                # Filter out rows with masked jersey numbers
                self.detection_df = self.detection_df.loc[
                    ~(
                        (self.detection_df["match_key"] == match_key)
                        & (
                            self.detection_df["jersey_number_gt"].isin(
                                self.masked_numbers
                            )
                        )
                    )
                ]

                total_masked_samples += masked_count
                self.logger.info(
                    f"Removed {masked_count} samples with masked numbers in {match_key}"
                )

            # Update possible numbers for this match by removing masked numbers
            original_possible = data.possible_numbers
            data.possible_numbers = original_possible - self.masked_numbers

            # Get all jersey numbers in this match for tracking (after filtering)
            match_rows_after = self.detection_df[
                self.detection_df["match_key"] == match_key
            ]
            match_jerseys[match_key] = set(
                match_rows_after.loc[
                    match_rows_after["jersey_number_gt"].notna(), "jersey_number_gt"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

        # Log masking statistics
        if total_masked_samples > 0:
            self.logger.info(
                f"Total samples removed: {total_masked_samples} ({total_masked_samples / original_size * 100:.2f}% of dataset)"
            )
            self.logger.info(
                f"Dataset size reduced from {original_size} to {len(self.detection_df)} samples"
            )

            # Log which numbers are still present in the dataset
            remaining_numbers = set()
            for numbers in match_jerseys.values():
                remaining_numbers.update(numbers)
            self.logger.info(
                f"Numbers present in dataset after masking: {sorted(remaining_numbers)}"
            )
