# Grain Disease Classification: Image Recognition MCO

A machine learning project for automated grain quality and disease classification using supervised learning models. This project implements and compares three different algorithms for classifying rice and maize kernel images.

**Course:** Advanced Intelligent Systems (STINTSY)  
**Team:** Dataset Divas (S04)  
**Members:**
- Casas, Ana Gabrielle Luis D.
- Esponilla, Ellexandrei A.
- Filipino, Eunice Marble R.
- Punsalan, Emmanuel Gerald G.

## Project Overview

This Major Course Output (MCO) focuses on automating grain quality control through image classification. By training multiple supervised learning models on high-resolution kernel images, the project aims to support agricultural automation and food security—with a modern twist toward nutritional wellness.

### Why This Project Matters

Grains are a cultural and nutritional backbone of the Philippines. Automating grain quality control revolutionizes the agricultural sector by ensuring consistent quality, supporting national food security, and enabling efficient processing.

## Dataset

The project uses the **[GrainSet Dataset](https://www.nature.com/articles/s41597-023-02660-8)**, a large-scale, expert-annotated image collection for visual quality inspection of cereal grains.

### Dataset Details
- **Total Images:** 350,000+ single-kernel images
- **Grain Types in This Project:** Rice and Maize only
- **Image Capture:** Custom high-resolution imaging system
- **Annotations:** Expert-labeled damage categories

### Disease/Condition Categories

**Maize Categories (7 classes):**
- `0_NOR` - Normal
- `1_F&S` - Fungal & Spot
- `2_SD` - Sterile Dwarf
- `3_MY` - Maize Yeast
- `4_AP` - Aflatoxin Positive
- `5_BN` - Brown Necrosis
- `6_HD` - Head Damage

**Rice Categories (7 classes):**
- `0_NOR` - Normal
- `1_F&S` - Fungal & Spot
- `2_SD` - Sterile Dwarf
- `3_MY` - Mold Yellowing
- `4_AP` - Aflatoxin Positive
- `5_BN` - Brown Necrosis
- `6_UN` - Unripe

## Project Structure

```
.
├── Notebooks (Sequential Workflow)
│   ├── 01_intro_and_data_preparation.ipynb    # Data integrity verification
│   ├── 02_exploratory_data_analysis.ipynb     # Dataset exploration & visualization
│   ├── 03_preprocessing.ipynb                  # Data augmentation & validation splits
│   ├── 04_logistic_regression.ipynb           # Logistic Regression model training
│   ├── 05_svm.ipynb                            # SVM model training
│   ├── 06_neural_network.ipynb                # Deep Learning model training
│   └── 07_comparisons.ipynb                    # Model comparison & evaluation
│
├── Python Modules
│   ├── preprocessing.py                        # Preprocessing utilities & data augmentation
│   ├── logistic_regression_lib.py             # Logistic Regression implementation
│   ├── neural_network_utils.py                # Neural Network utilities
│
├── Data
│   ├── dataset/
│   │   ├── images/
│   │   │   ├── maize/  (train/val/test splits with 7 disease categories)
│   │   │   └── rice/   (train/val/test splits with 7 disease categories)
│   │   ├── maize.xml                          # Maize annotations
│   │   └── rice.xml                           # Rice annotations
│
├── Models & Checkpoints
│   ├── checkpoints/                           # Saved model checkpoints
│   ├── models/                                # Trained model outputs
│   └── lightning_logs/                        # PyTorch Lightning logs
│
└── Additional Files
    ├── logs/                                  # Training metrics & logs
    ├── outputs/                               # Model predictions & results
    └── Poster - Dataset Divas - STINTSY - S04.pdf
```

## Models Implemented

### 1. Logistic Regression
A baseline linear classifier using HOG (Histogram of Oriented Gradients) feature extraction for efficient classification.

### 2. Support Vector Machine (SVM)
A non-linear classifier utilizing kernel methods for improved classification accuracy over logistic regression.

### 3. Neural Network / Deep Learning
A convolutional neural network (potentially with transfer learning using EfficientNet) for state-of-the-art image classification performance.

## Installation & Setup

### Prerequisites
- Python 3.12+
- Conda (recommended)

### Environment Setup

```bash
# Activate conda environment
conda activate python3.12

# Install required packages
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch / Lightning
- scikit-learn
- OpenCV (cv2)
- Pillow
- NumPy, Pandas
- Albumentations (data augmentation)
- Matplotlib, Seaborn (visualization)

## Workflow

1. **Data Preparation** (`01_intro_and_data_preparation.ipynb`)
   - Verify image integrity
   - Handle corrupted files
   - Organize dataset structure

2. **Exploratory Data Analysis** (`02_exploratory_data_analysis.ipynb`)
   - Dataset statistics and distribution
   - Visualize sample images
   - Analyze class balance

3. **Preprocessing** (`03_preprocessing.ipynb`)
   - Image normalization and resizing (letterbox method)
   - Data augmentation (rotation, flipping, color jitter, etc.)
   - Create validation splits

4. **Model Training**
   - Logistic Regression (`04_logistic_regression.ipynb`)
   - SVM (`05_svm.ipynb`)
   - Neural Network (`06_neural_network.ipynb`)

5. **Model Comparison** (`07_comparisons.ipynb`)
   - Evaluate all models
   - Compare accuracy, precision, recall, F1-score
   - Visualize confusion matrices

## Running the Project

```bash
# Navigate to project directory
cd image-recognition-mco

# Run notebooks sequentially
jupyter notebook

# Or open in VS Code with Jupyter support
code .
```

## Results & Outputs

- **Training Metrics:** Stored in `logs/tuned-efficientnet/`
- **Model Checkpoints:** Saved in `checkpoints/`
- **Predictions & Analysis:** Generated in `outputs/`

## Key Preprocessing Techniques

- **Letterbox Padding:** Maintains aspect ratio by padding to 1:1 while preserving image content
- **Data Augmentation:** Rotation, flipping, color jitter, and other transformations to prevent overfitting
- **Train/Validation/Test Split:** 70% / 15% / 15% split with stratification

## Feature Extraction Methods

- **HOG (Histogram of Oriented Gradients):** For traditional ML models
- **Transfer Learning:** EfficientNet backbone for neural networks
- **Custom CNN:** Convolutional architecture for direct image processing

## Performance Metrics

The project evaluates models using:
- Accuracy
- Precision & Recall (per class)
- F1-Score
- Confusion Matrices
- ROC-AUC curves

## Future Enhancements

- Ensemble methods combining all three models
- Real-time inference pipeline
- Mobile deployment for field use
- Additional grain types (wheat, sorghum)
- Explainability analysis (LIME, SHAP)

## References

- GrainSet Dataset: https://www.nature.com/articles/s41597-023-02660-8
- PyTorch Lightning: https://lightning.ai/
- Albumentations: https://albumentations.ai/
- EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch

## Notes

- Dataset is sourced from the GrainSet Dataset (see link above)
- All notebooks should be run sequentially for proper workflow
- GPU acceleration recommended for neural network training
- Reproducibility ensured through random state seeds

---

*This project was completed as part of the Advanced Intelligent Systems course (STINTSY) requirement.*
