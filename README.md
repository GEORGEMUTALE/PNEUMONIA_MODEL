<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Research_Complete-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

# Pediatric Pneumonia Detection from Chest X-rays
### *Binary Classification: Normal vs. Pneumonia*

<br>

## 📖 Overview
This project develops a **deep learning-based screening system** to distinguish **normal** from **pneumonia-affected** chest radiographs in children under five. Pneumonia remains a leading cause of pediatric mortality worldwide, and an automated, consistent screening tool could provide significant clinical value—particularly in resource-limited settings where specialist readers are scarce.

Unlike a simple black-box classifier, this system incorporates **interpretability via Grad-CAM** to visualize the regions of the X-ray that drive the model's decision, building clinical trust by demonstrating that the model attends to the lung fields rather than irrelevant image markers.

---

## 🧠 Model Architecture & Strategy
We leverage **Transfer Learning** to overcome the limited size of the pediatric chest X-ray dataset. Two state-of-the-art backbones were compared, with **DenseNet121** ultimately selected for its superior feature reuse and generalization.

- **DenseNet121**: Pre-trained on ImageNet, fine-tuned on our chest X-ray dataset.
- **ResNet18** (11.2M parameters): Lighter and faster, but less powerful—tested for comparison.
- **Custom Classifier Head**: `Dropout(0.5) → Linear(128) → ReLU → Dropout(0.5) → Linear(2)`.
- **Optimization**: AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).
- **Exponential Moving Average (EMA)**: Applied to model weights for stable, smooth validation performance (contributed ~3% accuracy gain).
- **Transfer Learning Strategy**: Early layers frozen; deeper blocks fine-tuned to adapt to radiographic features.
- **Threshold Tuning**: Decision boundary optimized on the validation set to achieve ≥95% sensitivity, prioritizing the detection of affected children.

---

## 📊 Dataset
We use the publicly available **Chest X-Ray Images (Pneumonia)** dataset (Mooney, 2018; Kermany et al., 2018), consisting of pediatric radiographs from children aged 1–5 years, collected at a medical center in Guangzhou, China.

| Class | Description | Training | Validation | Test |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | Healthy control radiographs | 1,140 | 281 | 234 |
| **Pneumonia** | Confirmed pneumonia cases | 3,289 | 522 | 390 |
| **Total** | | **4,429** | **803** | **624** |

> **Note**: The dataset ships with only 16 validation images (far too few). We discarded this folder and constructed a **patient-aware partition** to prevent data leakage, ensuring no patient appears in more than one subset.

### ⚠️ Key Preprocessing Decisions
- **Patient-Aware Splitting**: All images from a single patient are confined to one subset (train/val/test) to prevent the model from recognizing individuals rather than disease.
- **Horizontal Flip Excluded**: Mirroring a chest X-ray would place the heart on the anatomically incorrect side—a pattern that never occurs in real clinical images.
- **Deduplication**: The raw dataset contained repetitive sub-revisions (multi-source "Kinderweg" variants); a perceptual hashing protocol was applied to prevent data leakage.
- **Class Weighting**: The minority class (Normal) was weighted more heavily in the loss function to counteract the 74% pneumonia imbalance.

---

## ⚙️ Key Technical Features

### 🔄 Online vs. Offline Augmentation
*This project systematically tested both strategies, concluding that **dynamic online augmentation** significantly outperforms static offline scaling.*

- **Online Augmentation (Final Model)**: Random brightness/contrast shifts, rotations (±10°), translations, and random cropping (85%-100%). Applied stochastically at *every* epoch to provide infinite variations.
- **Offline Benchmark (Controlled Test)**: A static 4× expansion was tested, bloating the dataset from **4 GB to 15 GB**. It revealed:
  - **Static Memorization**: The model memorized fixed noise patterns, causing a sharp validation loss spike after epoch 15.
  - **I/O Starvation**: GPU utilization dropped significantly due to CPU-to-GPU data transfer bottlenecks.
  - **Performance Drop**: AUC dropped from **0.978 to 0.938**.
- **Noise Experiment**: Adding Gaussian and salt-and-pepper noise *harmed* performance (AUC 0.978 → 0.938) and was removed from the final pipeline.

### 🖥️ Infrastructure Optimization
- **GPU**: T4x2 (dual GPUs with 16GB VRAM each), significantly outperforming legacy P4 GPUs.
- **Batch Tuning**: Batch sizes dynamically adjusted (32 for offline, 48 for online) to manage memory constraints.
- **I/O Management**: Implemented asynchronous data prefetching and gradient accumulation to mitigate severe I/O starvation.
- **Kaggle Constraints**: The 15 GB offline dataset repeatedly triggered 9-hour session timeouts, forcing early stopping and checkpoint monitoring.

---

## 📈 Results & Performance

The best configuration (**DenseNet121 + EMA + Online Augmentation**) achieved outstanding discriminative performance on the held-out test set.

| Threshold | Accuracy | Sensitivity | Specificity | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Default (0.50)** | **90.0%** | **98.7%** | **75.2%** | **0.978** |
| **Tuned (0.80)** | **93.3%** | **98.5%** | **84.6%** | **0.978** |

> **Statistical Validation**: A chi-square test on the 2×2 confusion matrix yielded a highly significant association (**χ² = 388.8, df = 1, p < 0.0001**), confirming the predictions are far from random.

### 🔬 Ablation Studies
To isolate the effect of individual engineering decisions, we conducted the following comparisons:

| Configuration Variation | Test Accuracy | ROC-AUC | Observation |
| :--- | :--- | :--- | :--- |
| **Baseline (DenseNet121 + Online)** | **90.0%** | **0.978** | Optimal generalization. |
| + Offline Static Augmentation (15GB) | 82.5% | 0.938 | **Severe drop** due to memorization & I/O bottlenecks. |
| + Synthetic Noise (Gaussian/S&P) | 82.5% | 0.938 | Obscured fine pathological features; removed. |
| - Exponential Moving Average (EMA) | 87.2% | 0.966 | EMA contributed ~3% accuracy gain. |
| ResNet18 (vs. DenseNet121) | 86.9% | 0.968 | Lighter & faster, but less powerful. |
| Training for 100 epochs (vs. 50) | 90.2% | 0.977 | Negligible gain; cheaper schedule retained. |

---

## 🔍 Model Interpretability (Grad-CAM)
To ensure clinical trust, we implemented **Grad-CAM** heatmaps (Selvaraju et al., 2017). The visualizations confirm that the model focuses its attention on the **lung fields and central chest regions**—the exact areas a radiologist inspects—rather than irrelevant image markers or borders.

> When the model makes an error, the Grad-CAM activation frequently drifts toward the diaphragm or image corners, providing a useful warning mechanism that the prediction may be unreliable.

<p align="center">
  <img src="assets/gradcam_sample.png" alt="Grad-CAM Heatmaps" width="600">
  <br>
  <em>Figure: Grad-CAM overlays showing model attention on lung regions.</em>
</p>

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
