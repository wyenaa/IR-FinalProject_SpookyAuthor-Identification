# 👻 Spooky Author Identification: NLP Model Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project aims to identify the author of horror stories based on text excerpts. Using the [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification) dataset from Kaggle, the model predicts whether a given text was written by:
1.  **Edgar Allan Poe (EAP)**
2.  **H.P. Lovecraft (HPL)**
3.  **Mary Shelley (MWS)**

The goal is to minimize **Multi-class Logarithmic Loss (Log Loss)**. The project compares a strong Classical Machine Learning Baseline against a Deep Learning approach.

## Methodology & Experiments

### 1. Classical Machine Learning (The Robust Baseline)
We implemented a **Voting Classifier Ensemble (Soft Voting)** combining three distinct algorithms to create a strong baseline:
* **Feature Engineering:** TF-IDF Vectorization + Meta Features (Word count, Stopword count, Punctuation count).
* **Models:**
    * *Logistic Regression:* Captures linear relationships.
    * *Multinomial Naive Bayes:* Effective for word frequency analysis.
    * *SGD Classifier (Modified Huber):* Efficient optimization for high-dimensional text data.
* **Strategy:** Soft-voting average of probabilities from all three models.

### 2. Deep Learning (The Winning Approach) 
We designed a custom Neural Network architecture focused on efficiency and generalization, inspired by the **FastText** architecture.
* **Input:** Tokenized sequences with Padding.
* **Architecture:** Embedding Layer $\rightarrow$ Spatial Dropout $\rightarrow$ Global Average Pooling $\rightarrow$ Dense Layers.
* **Why this works:** Unlike complex RNNs/LSTMs that can overfit on small datasets, this architecture captures the "global sentiment/style" of the author by averaging word embeddings, making it highly effective for Stylometry tasks.

## Model Performance

The table below summarizes the validation results. The Deep Learning model successfully outperformed the Ensemble Baseline.

| Experiment Name | Model Architecture | Validation Log Loss | Status |
| :--- | :--- | :--- | :--- |
| **Baseline** | Ensemble (Logistic Reg + Naive Bayes + SGD) | `0.4767` | Strong Baseline |
| **Deep Learning** | **Embedding + SpatialDropout + GlobalAvgPool** | **`0.4246`** | **Winner** |

> *Note: Lower Log Loss indicates better performance and higher model confidence.*

## Deep Learning Architecture Details

The winning model uses the following structure:

```python
model = Sequential([
    # 1. Maps words to 100-dim vectors (Semantic Understanding)
    Embedding(input_dim=20000, output_dim=100, input_length=100),
    
    # 2. Drops entire 1D feature maps to prevent overfitting
    SpatialDropout1D(0.2),
    
    # 3. Averages vectors to capture global sentence 'style'
    GlobalAveragePooling1D(),
    
    # 4. Dense layer for non-linear pattern matching
    Dense(50, activation='relu'),
    Dropout(0.2),
    
    # 5. Output probabilities for 3 Authors
    Dense(3, activation='softmax')
])
