# Detection-of-AI-Generated-Arabic-Text-A-Data-Mining-Approach

## Overview
The rapid adoption of large language models has led to a growing amount of AI-generated text, raising concerns about authorship authenticity and content reliability. This project investigates the detection of AI-generated **Arabic** text using interpretable linguistic features and traditional machine learning models. The focus is on building a transparent and efficient classification framework that distinguishes between human-written and AI-generated academic abstracts.

---

## Dataset
The dataset used in this project is publicly available on Hugging Face:

**Arabic Generated Abstracts Dataset**  
https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts

The dataset consists of academic abstracts from multiple sources. After combining the four subsets, the data was labeled into a binary classification task:

- **AI-generated:** 33,552 samples  
- **Human-written:** 8,388 samples  

### Loading the Dataset
```python
from datasets import load_dataset
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")
