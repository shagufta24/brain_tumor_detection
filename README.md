# 🧠 Brain Tumor Detection in MRI Scans

A deep learning-based computer vision project that detects the presence of brain tumors from MRI scan images. The model classifies images into two categories: **Tumor** (`yes`) and **No Tumor** (`no`). This repository includes data preprocessing, augmentation, CNN-based modeling, transfer learning, and performance evaluation.

## 📁 Dataset

- **Source**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**:
  - `yes`: MRI scans showing presence of a brain tumor.
  - `no`: MRI scans showing absence of a brain tumor.

## 🛠️ Features

- Data loading and preprocessing.
- Brain region cropping using image processing techniques.
- Data augmentation using `albumentations` to expand dataset diversity.
- Custom CNN model as a baseline.
- Transfer learning with fine-tuned VGG16 and EfficientNet.
- Model evaluation with accuracy, loss, confusion matrix, and classification report.
- Visualization of training performance and feature maps.
  
---

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, Albumentations, Matplotlib, Seaborn, Scikit-learn
- **Tools**: Google Colab, Kaggle API
---

## ⚙️ How It Works

### 1. **Data Preprocessing**
- Cropped unnecessary black areas from MRIs to focus on the brain region.
- Images resized to **224x224** pixels.
- Data split: `80%` Train, `10%` Validation, `10%` Test.

### 2. **Data Augmentation**
- Applied techniques: Horizontal/Vertical Flip, Random Rotation, Affine Transformations, Elastic Transform, Brightness/Contrast adjustment, Gaussian Noise.
- Augmented dataset balanced at 500 images per class.

### 3. **Baseline CNN Model**
- Simple architecture with 3 convolutional blocks and dropout regularization.
- Activation function: ReLU.
- Output: Sigmoid activation for binary classification.
- Loss: Binary Crossentropy.
- Optimizer: Adam (`learning_rate=1e-4`).

### 4. **Transfer Learning with VGG16**
- Pre-trained VGG16 (ImageNet weights) with custom classifier layers.
- Fine-tuning last 5 layers for feature extraction.
- Added GlobalAveragePooling, Dense layers, and Dropout.

### 5. **Transfer Learning with EfficientNetB0**
- Pre-trained EfficientNetB0 model (ImageNet weights) as a feature extractor.
- Added GlobalAveragePooling, Dense layers, and Dropout for classification.
- Fine-tuned last few layers for improved performance on the brain MRI dataset.

### 6. **Training & Evaluation**
- EarlyStopping and ReduceLROnPlateau callbacks.
- Visualized learning curves (Accuracy & Loss).
- Confusion matrix and classification reports for Test and Validation sets.
