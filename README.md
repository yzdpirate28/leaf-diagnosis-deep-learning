# 🌿 Plant Disease Detection using CNN | Data Mining Project 2025

This repository contains a full end-to-end data mining pipeline for **image-based plant disease classification**, completed as part of the DATA2025 course. Using deep learning and the Plant Village dataset, we built a CNN to classify crop diseases across 38 categories, with deployment-ready outputs.

---

## 📦 Dataset

- **Name:** Plant Village (Augmented Version)
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size:** ~87,000 RGB images across 38 classes
- **Type:** Image Classification
- **Split:** 80% Train / 20% Validation / 33 standalone test images

---

## 🛠️ Features of the Project

- 📚 Data cleaning (check for corrupted and duplicate images)
- 🧠 Built a CNN model using TensorFlow and Keras
- 📊 Exploratory Data Analysis and visualization
- 🧪 Evaluated model with accuracy, confusion matrix, classification report
- 💾 Model saved and reused for prediction
- 📷 Real-time testing on unseen images
- 🖼️ Visualization of model predictions on test samples

---

## 🧠 Model Architecture (CNN)

- 5 Convolutional Blocks (with MaxPooling)
- Flatten → Dense (1500 units) → Dropout
- Output Layer: 38 softmax units
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical Crossentropy
- Accuracy: Trained up to **~99%** accuracy on validation set

---

## 📊 Performance

| Dataset     | Accuracy |
|-------------|----------|
| Training    | ~99%     |
| Validation  | ~99%     |
| Testing     | Real-time image predictions successful |

Confusion matrix and classification report included for all classes.

---

## 🖼️ Sample Predictions

![Prediction Example](Plant_Disease_Dataset/test/test/PotatoHealthy1.JPG)  

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yzdpirate/leaf-diagnosis-deep-learning.git
cd leaf-diagnosis-deep-learning

<pre> ```leaf-diagnosis-deep-learning/
├── dataminingenv/
├── notebooks/
│   └── test_notebook.ipynb
│   └── train_notebook.ipynb
├── src/
│   ├── test
│   └── train
│   └── valid
├── .gitignore
├── README.md
├── requirements.txt
└── trained_model.keras``` </pre>
