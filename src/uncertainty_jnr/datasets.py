"""Dataset definitions and utilities."""

from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel

from uncertainty_jnr.data import JerseyNumberDataset
from torch.utils.data import ConcatDataset


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    match_ids: List[str]
    params: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


VALIDATION_DATASETS = {
    "ca": DatasetConfig(
        match_ids=[  # Copa America 3 matches sample
            "argentina-brazil-20210711",
            "venezuela-canada-20240706",
            "uruguay-colombia-20240711",
        ]
    ),
    "ca12": DatasetConfig(
        match_ids=[  # Copa America 12 matches sample
            "argentina-brazil-20210711",
            "venezuela-canada-20240706",
            "uruguay-colombia-20240711",
            "argentina-canada-20240621",
            "brazil-costa_rica-20240625",
            "colombia-paraguay-20240624",
            "ecuador-venezuela-20240622",
            "mexico-jamaica-20240623",
            "peru-chile-20240622",
            "united_states-bolivia-20240623",
            "uruguay-panama-20240624",
            "argentina-colombia-20240715",
        ]
    ),
    "soccernet_test": DatasetConfig(match_ids=["soccernet_test"]),
    "soccernet_test_sample": DatasetConfig(
        match_ids=["soccernet_test"], params={"limit_samples": 50000}
    ),
    "soccernet_challenge": DatasetConfig(
        match_ids=["soccernet_challenge"], params={"inference_mode": True}
    ),
    "soccernet_challenge_sample": DatasetConfig(
        match_ids=["soccernet_challenge"],
        params={"limit_samples": 30000, "inference_mode": True},
    ),
    "reid_valid": DatasetConfig(
        match_ids=["soccernet_reid_valid"]
    ),  # SoccerNet ReID valid split
    "reid_train": DatasetConfig(
        match_ids=["soccernet_reid_train"]
    ),  # SoccerNet ReID train split
    "reid_test": DatasetConfig(
        match_ids=["soccernet_reid_test"]
    ),  # SoccerNet ReID test split
    "reid_challenge": DatasetConfig(
        match_ids=["soccernet_reid_challenge"]
    ),  # SoccerNet ReID challenge split
}

TRAIN_DATASETS = {
    "reid_all": DatasetConfig(
        match_ids=[
            "soccernet_reid_train",
            "soccernet_reid_valid",
            "soccernet_reid_test",
            "soccernet_reid_challenge",
        ]
    ),
    "soccernet_train_full": DatasetConfig(match_ids=["soccernet_train"], params={}),
    "soccernet_train": DatasetConfig(
        match_ids=["soccernet_train"], params={"limit_samples": 100000}
    ),
}


def extract_teams_from_match_id(match_id: str) -> tuple[str, str]:
    """Extract team names from match ID.

    Args:
        match_id: Match ID in format "team1-team2-date"

    Returns:
        Tuple of team names (team1, team2)
    """
    if not match_id or "-" not in match_id:
        return "", ""

    parts = match_id.split("-")
    if len(parts) < 3:
        return "", ""

    return parts[0], parts[1]


def create_datasets(
    config,
    val_dataset_names: List[str],
    train_dataset_names: List[str],
    masked_numbers: Optional[List[int]] = None,
) -> tuple[Union[ConcatDataset, JerseyNumberDataset], Dict[str, JerseyNumberDataset]]:
    """Create train and validation datasets.

    Args:
        config: Configuration object
        val_dataset_names: List of validation dataset names
        train_dataset_names: List of training dataset names
        masked_numbers: Optional list of jersey numbers to mask during training

    Returns:
        Tuple of (train_dataset, val_datasets_dict)
    """
    from uncertainty_jnr.augmentation import get_train_transforms, get_val_transforms

    # Get all validation matches
    all_val_matches = []
    for val_name in val_dataset_names:
        if val_name not in VALIDATION_DATASETS:
            raise ValueError(f"Validation dataset {val_name} not found")
        all_val_matches.extend(VALIDATION_DATASETS[val_name].match_ids)

    # Extract teams from validation matches
    val_teams = set()
    for match in all_val_matches:
        team1, team2 = extract_teams_from_match_id(match)
        if team1 and team2:
            val_teams.add(team1)
            val_teams.add(team2)

    # Create train datasets
    train_datasets = []
    for train_dataset_name in train_dataset_names:
        if train_dataset_name not in TRAIN_DATASETS:
            raise ValueError(f"Training dataset {train_dataset_name} not found")

        dataset_config = TRAIN_DATASETS[train_dataset_name]
        train_matches_pre_filter = dataset_config.match_ids
        train_matches = []

        # Filter out matches with teams in validation set
        for match in train_matches_pre_filter:
            team1, team2 = extract_teams_from_match_id(match)
            if team1 in val_teams or team2 in val_teams:
                continue
            train_matches.append(match)

        # Get default dataset parameters
        dataset_params = {
            "root_dir": config.data.root_dir,
            "match_ids": train_matches,
            "target_size": config.data.target_size,
            "cache_dir": config.data.cache_dir,
            "transform": get_train_transforms(
                target_size=config.data.target_size,
                interpolation_method=config.data.interpolation_method,
                p_scale=config.data.train_aug_p_scale,
                p=config.data.train_aug_p,
            ),
            "filter_invalid": True,
            "masked_numbers": masked_numbers,
        }

        # Override with dataset-specific parameters if provided
        if dataset_config.params:
            dataset_params.update(dataset_config.params)

        # Create dataset
        train_dataset = JerseyNumberDataset(**dataset_params)
        train_datasets.append(train_dataset)

    # Combine train datasets if multiple
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = train_datasets[0]

    # Create validation datasets
    val_datasets = {}
    for val_name in val_dataset_names:
        val_dataset_config = VALIDATION_DATASETS[val_name]

        # Get default dataset parameters
        dataset_params = {
            "root_dir": config.data.root_dir,
            "match_ids": val_dataset_config.match_ids,
            "target_size": config.data.target_size,
            "cache_dir": config.data.cache_dir,
            "transform": get_val_transforms(
                target_size=config.data.target_size,
                interpolation_method=config.data.interpolation_method,
            ),
            "filter_invalid": True,
        }

        # Override with dataset-specific parameters if provided
        if val_dataset_config.params:
            dataset_params.update(val_dataset_config.params)

        # Create dataset
        val_datasets[val_name] = JerseyNumberDataset(**dataset_params)

    return train_dataset, val_datasets
