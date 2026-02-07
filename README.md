# Biometric Defence Against Realistic Presentation Attacks Using rPPG

This project implements a CNN classifier for detecting presentation attacks using remote Photoplethysmography (rPPG) signals. The system analyzes PPG spectral maps to distinguish between genuine biometric samples and presentation attacks.

## Project Overview

Presentation attacks (PAs) are security threats where attackers use fake biometric samples (like photos, videos, or 3D masks) to bypass biometric authentication systems. This project uses rPPG signals to detect these attacks by analyzing subtle physiological signals that are difficult to fake.

## Dataset

This project uses the **ROSE-Youtu Face Liveness Detection Database** for training and evaluation. The original video sequences were processed to extract rPPG signals, which were then converted to spectral maps for CNN classification.

### Dataset Access:
**Note**: The ROSE-Youtu dataset is not publicly available. You will need to:
1. **Request access** from the official ROSE-Youtu database maintainers
2. **Process your own data** using the provided preprocessing scripts
3. **Use your own dataset** with similar format

### Data Format:
The code expects preprocessed data in the following format:
- **File**: `ppg_spec_maps` (pickle file)
- **Content**: (X, y) where X are PPG spectral maps and y are labels
- **Shape**: X should be (N, 10, 31, 1) where N is number of samples
- **Labels**: Binary classification (0 = genuine, 1 = attack)

## Methodology

### rPPG Signal Processing
- Remote Photoplethysmography: Non-contact measurement of blood volume changes
- Spectral Analysis: Conversion of temporal signals to frequency domain
- CNN Classification: Deep learning approach for attack detection

### Model Architecture
- Input: PPG spectral maps (10×31×1)
- Convolutional Layers: 3 layers with increasing filters (32→64→128)
- Regularization: Dropout layers (0.5-0.7) to prevent overfitting
- Output: Binary classification (genuine vs. attack)

## Results

- Accuracy: ~96% on validation set
- Precision: 0.95-0.98 for both classes
- Recall: 0.94-0.98 for both classes
- F1-Score: 0.96 for both classes

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Setup
1. **For ROSE-Youtu users**: Request access and process videos to extract rPPG signals
2. **For other datasets**: Process your data to match the expected format
3. **Place your data**: Put `ppg_spec_maps` in the `data/` directory
4. **Run preprocessing**:
```bash
python src/data_preprocessing.py
```

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/inference.py --input_path path/to/test/data
```

## Project Structure

```
biometric-defense-rppg/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
├── notebooks/
│   └── cnn_classifier_of_ppg_maps.ipynb  # Original Jupyter notebook
├── src/
│   ├── data_preprocessing.py           # Data loading and preprocessing
│   ├── model.py                        # CNN model definition
│   ├── train.py                        # Training script
│   └── inference.py                    # Prediction script
├── data/
│   └── README.md                       # Dataset documentation
├── results/
│   ├── Output Screenshot.pdf           # Project output screenshots
│   └── Project Result .pdf             # Project results documentation
└── models/                             # Trained models (created during training)
```

## Usage Examples

### Training the Model
```python
from src.model import create_cnn_model
from src.data_preprocessing import load_data

# Load data
X_train, X_val, y_train, y_val = load_data()

# Create and train model
model = create_cnn_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

### Making Predictions
```python
from src.inference import predict_attack

# Predict on new data
prediction = predict_attack("path/to/test_data")
print(f"Attack probability: {prediction}")
```

## Experiments

The model was trained with the following parameters:
- Epochs: 200
- Batch Size: 32
- Optimizer: Adam
- Loss: Binary Crossentropy
- Data Augmentation: Rotation, shifts, flips, zoom

## References

1. Remote Photoplethysmography (rPPG) for Biometric Authentication
2. CNN-based Presentation Attack Detection
3. Spectral Analysis in Biometric Security
4. ROSE-Youtu Face Liveness Detection Database

## Authors

- SURIYA M & DEEPAK DEVAKUMAR S

## Acknowledgments

- ROSE-Youtu Face Liveness Detection Database
- Research community for rPPG signal processing techniques
- Open-source deep learning frameworks

## Contact

- Email: suriyamurugavel2005@gmail.com
