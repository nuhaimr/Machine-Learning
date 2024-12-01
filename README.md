# Coral Images Classification Project - README

## Overview

This project focuses on building machine learning models for classifying coral images using deep learning techniques. The repository contains Jupyter notebooks for training and evaluating various pre-trained models and custom architectures. It also includes fine-tuning experiments and an analysis of hyperparameter choices.

---

## Contents

### Files and Notebooks

- **`alexnet_fineturnning.ipynb`**: Implementation of AlexNet with fine-tuning for coral image classification.
- **`resnet18_fineturnning.ipynb`**: Implementation of ResNet-18 with fine-tuning and results for coral classification.
- **`resnet50_fineturning(KNN).ipynb`**: Experimentation with ResNet-50 using fine-tuning and a K-Nearest Neighbors classifier.
- **`resnet50_fineturning.ipynb`**: Fine-tuning ResNet-50 for coral image classification.
- **`resnet50_scratch.ipynb`**: Training ResNet-50 from scratch for coral images.
- **`vgg16_fineturnning.ipynb`**: Fine-tuning VGG-16 for coral image classification tasks.
- **`Imran_Nuha_20696366.pdf`**: Detailed project report explaining model choices, hyperparameter tuning, data augmentation, and results.
- **`README.md`**: Project documentation (this file).

---

## Tasks and Objectives

1. **Task 1: Model Selection and Fine-Tuning**
   - Choose pre-trained models like ResNet, AlexNet, and VGG.
   - Apply fine-tuning techniques to improve model performance on coral images.
   - Experiment with hyperparameter tuning (learning rate, batch size, etc.).

2. **Task 2: Training from Scratch**
   - Train ResNet-50 from scratch to evaluate the performance compared to fine-tuned models.

3. **Task 3: Comparative Analysis**
   - Compare the results of fine-tuned models against those trained from scratch.
   - Evaluate different architectures based on accuracy, loss, and computational efficiency.

---

## Key Highlights

1. **Data Augmentation**:
   - Techniques used: Random rotation, flipping, cropping, and scaling.
   - Goal: Improve model robustness and generalization.

2. **Fine-Tuning**:
   - Modify and retrain specific layers of pre-trained models to adapt to coral images.
   - Models used: ResNet-50, ResNet-18, AlexNet, and VGG-16.

3. **Hyperparameter Tuning**:
   - Learning rates, optimizers, batch sizes, and dropout rates were optimized for performance.

4. **KNN Integration**:
   - Experimented with ResNet-50 and a KNN classifier to explore alternative approaches to classification.

5. **Performance Evaluation**:
   - Metrics: Accuracy, precision, recall, F1-score.
   - Tools: Confusion matrices and learning curves.

---

## Usage Instructions

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/coral-images-classification.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the desired notebook and run the cells.

---

## Results Summary

### Key Findings
1. **Fine-Tuned Models**:
   - ResNet-50 and ResNet-18 showed the highest accuracy and robustness with fine-tuning.
   - VGG-16 and AlexNet required more epochs to achieve comparable results.

2. **Training from Scratch**:
   - Models trained from scratch showed lower performance compared to fine-tuned models, demonstrating the importance of transfer learning for small datasets.

3. **Data Augmentation**:
   - Improved generalization and helped mitigate overfitting.

---

## References
- Detailed explanations and results are available in the project report: `Imran_Nuha_20696366.pdf`.

---

## Contact
For any questions or issues, please contact the repository maintainer.

---

*Explore coral image classification with state-of-the-art models!* 🌊
