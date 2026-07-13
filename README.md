<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

# 🫁 Multi-Class Pneumonia & COVID-19 Classifier
### *Normal vs. COVID-19 vs. Bacterial Pneumonia vs. Viral Pneumonia*

<br>

## 📖 Overview
This project builds a **state-of-the-art deep learning pipeline** to classify pediatric and adult chest X-rays into **four distinct categories**:

1.  **Normal** (Healthy lungs)
2.  **COVID-19** (Viral infection caused by SARS-CoV-2)
3.  **Bacterial Pneumonia** (Typically lobar/segmental consolidation)
4.  **Viral Pneumonia** (Non-COVID viral infections, e.g., influenza, adenovirus)

Unlike a simple binary disease detector, this system outputs **probabilistic predictions** for each of the four classes, allowing clinicians to assess the model's confidence in its differential diagnosis. The top-1 predicted class is accompanied by its specific probability score, providing a transparent, explainable clinical decision support tool.

---

## 🧠 Model Architecture & Strategy
We leverage **Transfer Learning** to overcome the limited size of medical imaging datasets. Two state-of-the-art backbones were compared, with **DenseNet121** ultimately selected for its superior feature reuse and generalization.

- **DenseNet121**: Pre-trained on ImageNet, fine-tuned on our chest X-ray dataset.
- **Custom Classifier Head**: `Dropout(0.5) → Linear(128) → ReLU → Dropout(0.5) → Linear(4)`.
- **Optimization**: AdamW optimizer with decoupled weight decay.
- **Exponential Moving Average (EMA)**: Applied to model weights for stable, smooth validation performance.
- **Threshold Tuning**: Decision boundary optimized on the validation set to maximize sensitivity while maintaining high specificity across all 4 classes.

---

## 📊 Dataset
We utilize a publicly available collection of chest X-ray images merged from multiple institutional sources, ensuring a diverse patient population.

| Class | Description | Sample Count |
| :--- | :--- | :--- |
| **Normal** | Healthy control radiographs | ~1,500 |
| **COVID-19** | RT-PCR confirmed COVID-19 patients | ~1,500 |
| **Bacterial Pneumonia** | Confirmed bacterial lung consolidations | ~2,500 |
| **Viral Pneumonia** | Non-COVID viral infections (e.g., Influenza) | ~1,500 |

> **Preprocessing**: All images are resized to 224×224, normalized to ImageNet statistics, and augmented on-the-fly.

---

## ⚙️ Key Technical Features

### 🔄 Online vs. Offline Augmentation
*This project systematically tested both strategies, concluding that **dynamic online augmentation** significantly outperforms static offline scaling.*

- **Online Augmentation**: Random brightness/contrast shifts, rotations (±10°), translations, and random cropping (85%-100%). Applied stochastically at *every* epoch to provide infinite variations.
- **Offline Benchmark**: A static 4× expansion was tested to evaluate infrastructure trade-offs. It revealed severe I/O bottlenecks and overfitting, confirming that *more static data is not always better*.

### 🖥️ Infrastructure Optimization
- **GPU**: T4x2 (dual GPUs with 16GB VRAM each), significantly outperforming legacy P4 GPUs.
- **Batch Tuning**: Batch sizes dynamically adjusted between 32 and 48 depending on the pipeline's memory footprint.
- **I/O Management**: Implemented async data prefetching and gradient accumulation to mitigate the severe I/O starvation caused by large static datasets.

### 📈 Probabilistic Output
The model outputs a **softmax-activated 4-dimensional vector** for each image, representing the predicted probability distribution across the four classes. 
*Example:* `[0.01, 0.02, 0.95, 0.02]` → **Bacterial Pneumonia (95%)**.

---
