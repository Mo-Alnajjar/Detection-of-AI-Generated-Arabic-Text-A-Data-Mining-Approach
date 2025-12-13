# Detection-of-AI-Generated-Arabic-Text-A-Data-Mining-Approach

## Project Description
This project focuses on detecting **AI-generated Arabic text** using interpretable linguistic features and traditional machine learning models. Unlike deep neural approaches, this work emphasizes **feature-based analysis** to understand how linguistic patterns differ between human-written and AI-generated Arabic text.

The project uses academic abstracts as a case study and demonstrates that carefully designed linguistic features can effectively distinguish AI-generated content while remaining transparent and computationally efficient.

---

## Motivation
With the rapid spread of large language models, AI-generated text is becoming increasingly difficult to distinguish from human writing. This raises concerns in areas such as academic integrity, authorship verification, and content authenticity. These challenges are particularly significant for Arabic, a morphologically rich language with limited AI-text detection resources.

This project aims to:
- Build an interpretable detection system for Arabic AI-generated text
- Analyze which linguistic features are most informative
- Establish strong traditional ML baselines for future research

---

## Dataset
The dataset used in this project is publicly available on Hugging Face:

🔗 https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts

After combining four subsets and reformulating the task as a binary classification problem, the final class distribution is:

- **AI-generated:** 33,552 samples  
- **Human-written:** 8,388 samples  

Each sample contains normalized Arabic text and a binary label indicating whether the text is human-written or AI-generated.

---

## Preprocessing
The following preprocessing steps are applied:
- Arabic character normalization (e.g., Alif variants)
- Removal of diacritics and non-Arabic characters
- Whitespace normalization

These steps ensure consistent and clean input for linguistic analysis.

---

## Feature Engineering
Four linguistically motivated features are extracted:

- **Honoré’s R Measure**  
  Measures lexical richness and vocabulary sophistication.

- **Noun Count**  
  Counts nouns and proper nouns using Arabic POS tagging.

- **Genitive Construction Count**  
  Estimates idafa (genitive) structures via consecutive noun patterns.

- **Entity Density**  
  Computes the ratio of named entities to total words using Arabic NER.

These features were selected for their interpretability and relevance to Arabic writing style.

---

## Models
The project evaluates three traditional machine learning classifiers:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  

A stratified **70/15/15** train–validation–test split is used to ensure fair evaluation across classes.

---

## Evaluation
Models are evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

Experimental results show that **Random Forest achieves the most balanced performance**, particularly in handling class imbalance and capturing nonlinear relationships between features.

---

## Feature Importance
Analysis of the Random Forest model indicates that **entity density is the most influential feature**, suggesting that named entity usage patterns differ significantly between human-written and AI-generated Arabic text.

---

## Project Structure
```text
project/
├── data/
│   ├── raw/
│   │   └──            # Original, untouched dataset files
│   ├── processed/
│   │   └──            # Cleaned data and engineered feature files
│   └── external/
│       └──            # External or auxiliary datasets (if any)
│
├── src/
│   ├── data_preparation.py
│   │   └── Functions for text normalization, cleaning, and
│   │       linguistic feature extraction
│   │
│   ├── modeling.py
│   │   └── Model training, validation, testing, and evaluation
│   │       (Logistic Regression, SVM, Random Forest)
│   │
│   ├── visualization.py
│   │   └── Functions for plotting results, confusion matrices,
│   │       and feature importance charts
│   │
│   └── utils.py
│       └── Shared helper functions and reusable utilities
│
├── models/
│   └──                # Saved trained models (.pkl, .joblib)
│
├── notebooks/
│   └── analysis.ipynb # Exploratory analysis and experiments
│
├── requirements.txt   # Project dependencies
├── README.md          # Project description and usage
