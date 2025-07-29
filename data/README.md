# Dataset Information

## ROSE-Youtu Face Liveness Detection Database

This project uses the **ROSE-Youtu Face Liveness Detection Database** for training the rPPG-based biometric defense system.

### Dataset Access:
**Note**: The ROSE-Youtu dataset is not publicly available and is not included in this repository. You will need to:
1. **Request access** from the official ROSE-Youtu database maintainers
2. **Process your own data** to match the expected format
3. **Use alternative datasets** with similar characteristics

### Expected Data Format:
The code expects preprocessed data in the following format:
- **File**: `ppg_spec_maps` (pickle file)
- **Content**: (X, y) where X are PPG spectral maps and y are labels
- **Shape**: X should be (N, 10, 31, 1) where N is number of samples
- **Labels**: Binary classification (0 = genuine, 1 = attack)

### Files:
- `ppg_spec_maps`: Preprocessed PPG spectral maps (NOT INCLUDED - add your own)
- `preprocessed/`: Directory containing preprocessed data splits 

### Data Processing:
The original ROSE-Youtu video sequences were processed to extract rPPG signals, which were then converted to spectral maps for CNN classification.

### Usage:
1. Place your preprocessed data file in this directory
2. The dataset is automatically loaded and preprocessed by the training scripts

### Note:
This directory contains only documentation. You need to add your own preprocessed dataset file before running the training scripts. 