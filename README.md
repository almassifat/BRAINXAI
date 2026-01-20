# 🧠 BrainXAI

BrainXAI is a deep learning project that performs **EEG-based emotion recognition** using a **Multi-Layer Perceptron (MLP)** model and integrates **Explainable AI (XAI)** techniques to make predictions more interpretable.

This project is designed for experimenting with EEG signal features, training a neural network classifier, and analyzing model behavior using explainability tools.

---

## 📌 Project Overview

✅ **Goal:** Classify human emotions using EEG feature vectors  
✅ **Model:** Fully Connected Neural Network (MLP)  
✅ **Explainability:** XAI support (e.g., LIME / feature impact analysis)  
✅ **Framework:** PyTorch  
✅ **Dataset:** `emotions.csv`

---

## 🚀 Features

- EEG-based emotion classification
- PyTorch implementation of an MLP model
- Training & evaluation pipeline
- Loss and accuracy visualization
- Explainable AI (XAI) using **LIME**
- Clean and reproducible notebook workflow

---

## 🏗️ Model Architecture

The project uses an MLP architecture similar to:

- **Input Layer:** 2548 features  
- **Hidden Layer:** 64 neurons + ReLU  
- **Output Layer:** 3 emotion classes + Softmax  

📌 Model format: **(2548 → 64 → 3)**

---

## 📂 Repository Structure

```bash
BrainXAI/
│── BrainXAI_NET_Final.ipynb        # Main notebook (training + evaluation + XAI)
│── emotions.csv                    # EEG feature dataset
│── README.md                       # Project documentation
```

---

## ⚙️ Installation & Requirements

### ✅ Python Version
- Python **3.8+** recommended

### ✅ Required Libraries
Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn torch lime
```

---

## 🧾 Dataset

The dataset file used in this project:

📄 `emotions.csv`

It contains EEG feature vectors and corresponding emotion labels.

> Make sure `emotions.csv` is in the same folder as the notebook before running.

---

## ▶️ How to Run

### ✅ Option 1: Run in Jupyter Notebook / Colab

1. Open the notebook:
   ```bash
   BrainXAI_NET_Final.ipynb
   ```
2. Ensure dataset exists:
   - `emotions.csv`
3. Run all cells sequentially to:
   - load the dataset  
   - train the model  
   - evaluate results  
   - generate explanations using XAI  

---

## 📊 Training & Evaluation

During training, the model uses:

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Activation:** ReLU (hidden layers), Softmax (output layer)  

Evaluation includes:
- Accuracy measurement
- Performance visualization (loss/accuracy plots)

---

## 🔍 Explainable AI (XAI)

To make the model interpretable, BrainXAI integrates **LIME** (Local Interpretable Model-agnostic Explanations).

✅ XAI helps you understand:
- Which EEG features influenced a prediction
- Why the model predicted a specific emotion class
- Feature importance patterns across samples

---

## 🧠 Applications

BrainXAI can be useful for:

- Emotion-aware brain-computer interfaces (BCI)
- Mental health monitoring research
- Trustworthy AI in biomedical applications
- EEG-based classification benchmarking

---

## 🧑‍💻 Author

**Hasin Almas Sifat**  
📧 Email: hasin.almas.sifat@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/hasin-almas-sifat/  
💻 GitHub: https://github.com/almassifat  

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork or contribute!
