# Dataset Information

This document provides information about the datasets used in our research and their structure.

## Available Datasets

### SoccerNet ReID Dataset

This dataset contains pseudo-ground truth jersey number annotations for the SoccerNet ReID dataset based on our strongest proprietary model.

**Access:** The dataset is available through Google Drive:
- 📁 [SoccerNet ReID Dataset](https://drive.google.com/drive/folders/1m_bMmrBt5Q8-z1y0rRRRqgmfeBH-c97d?usp=sharing)

The download includes the following splits:
- `soccernet_reid_train.tar` (11.66 GB)
- `soccernet_reid_test.tar` (2.32 GB) 
- `soccernet_reid_valid.tar` (2.26 GB)
- `soccernet_reid_challenge.tar` (1.69 GB)

### Copa America (CA12) Dataset

Our proprietary Copa America dataset containing player tracking information together with ground truth jersey number annotations with very high precision.

**Access:** Please contact **Łukasz Grad** at **l.grad@mimuw.edu.pl** to request access for research purposes.

## Data Structure

Our datasets follow a unified structure:

```
{dataset}_{split}/
└── {match_part}/
    ├── detection/
    │   ├── track.csv                    # Detection-level tracking info
    │   └── jersey_number.csv            # Detection-level jersey predictions (baseline per-image predictions, contains visiblity information)
    ├── track/
    │   └── jersey_number_aggregated-annotated.csv  # Track-level GT jersey numbers
    ├── input_detections/
        └── {frame:06d}_{detection_id}.png  # Input 
```

### CSV File Formats

#### track.csv
Contains detection-level tracking information:
```csv
file_name,detection_id,track_id
000001,0,1234
000002,1,1234
...
```

#### jersey_number.csv
Contains detection-level jersey number annotations/predictions:
```csv
file_name,detection_id,jersey_number,score
000001,0,10,1.0
000002,1,10,0.95
...
```

#### jersey_number_aggregated-annotated.csv
Contains track-level ground truth jersey numbers:
```csv
track_id,jersey_number,jersey_number_score
1234,10,1.0
5678,7,1.0
...
```

## Notes

- **SoccerNet ReID**: Each image is treated as a single-frame track (no temporal information)
- **Copa America (CA12)**: Multi-frame tracks with additional challenging scenarios