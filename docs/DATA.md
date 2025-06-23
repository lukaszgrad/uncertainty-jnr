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
    │   └── jersey_number.csv            # Detection-level jersey predictions (baseline per-image predictions, contains visibility information)
    ├── track/
    │   └── jersey_number_aggregated-annotated.csv  # Track-level GT jersey numbers
    ├── input_detections/
        └── {file_name}_{detection_id}.png  # Input detection crops
```

### Common Column Definitions

- **`file_name`**: Frame number/identifier (integer) - uniquely identifies a video frame
- **`detection_id`**: Detection identifier within a frame (integer/float) - uniquely identifies a detection within that frame  
- **`track_id`**: Track identifier (integer) - links detections across frames belonging to the same player
- **`jersey_number`**: Jersey number annotation (0-99) - the actual jersey number worn by the player
- **`score`** / **`jersey_number_score`**: Confidence score (0.0-1.0) - indicates annotation quality/confidence

### Image Naming Convention

Detection crop images follow the pattern `{file_name}_{detection_id}.png`, where:
- `file_name` may be zero-padded (e.g., `000001`) or unpadded (e.g., `1`) 
- `detection_id` can be either integer or float format

This results in multiple possible naming patterns for the same detection:
- `000001_0.png` (padded file_name, integer detection_id)
- `1_0.png` (unpadded file_name, integer detection_id)  
- `000001_0.0.png` (padded file_name, float detection_id)
- `1_0.0.png` (unpadded file_name, float detection_id)

**Note:** A robust function `_get_possible_img_paths()` in the `data.py` module handles all these naming variations automatically when loading images.

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
Contains detection-level jersey number predictions from our baseline model:
```csv
file_name,detection_id,jersey_number,score
000001,0,10,1.0
000002,1,10,0.95
...
```

**Important Notes:**
- When an entry exists in this file, it means a pseudo-label from our baseline model exists for this detection crop
- The presence of an entry indicates the crop contains a **visible jersey number**
- ⚠️ **These labels are noisy** - they are predictions from our baseline model, not ground truth

#### jersey_number_aggregated-annotated.csv
Contains track-level ground truth jersey numbers:
```csv
track_id,jersey_number,jersey_number_score
1234,10,1.0
5678,7,1.0
...
```

**Important Notes:**
- These GT jersey numbers were **manually checked** (CA12 dataset only) and have **very high precision** of annotation
- `jersey_number_score` should be **1.0 at all times**  (for CA12 dataset) in this file (indicating perfect confidence in manual annotation)

## Notes

- **SoccerNet ReID**: Each image is treated as a single-frame track (no temporal information)
- **Copa America (CA12)**: Multi-frame tracks with additional challenging scenarios