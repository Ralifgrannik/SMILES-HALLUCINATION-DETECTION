# SMILES-2026 Hallucination Detection

This repository contains a lightweight binary classifier (probe) designed to detect hallucinations in the **Qwen2.5-0.5B** language model using its internal hidden states.

## Project Overview
The goal of this project is to distinguish between truthful (label 0) and hallucinated (label 1) model responses. By extracting and analyzing the hidden representations from the transformer layers, we identify patterns that correlate with model uncertainty and factual fabrication.

## Results
- **Primary Metric (Test AUROC):** 73.68%
- **Methodology:** L1-regularized Logistic Regression (Lasso) with 5-fold Stratified Cross-Validation.
- **Key Features:** Combined Mean/Max pooling from layers [14, 16, 18, 22] and residual drift (delta) analysis.

## Repository Structure
- `aggregation.py`: Feature extraction and layer-wise token pooling logic.
- `probe.py`: The `HallucinationProbe` classifier with embedded L1 feature selection.
- `splitting.py`: 5-fold Stratified K-Fold data splitting strategy.
- `solution.py`: Main entry point for extraction, training, and evaluation.
- `evaluate.py`: Core evaluation metrics and JSON reporting (fixed infrastructure).
- `results.json`: Final performance metrics averaged across folds.
- `SOLUTION.md`: Comprehensive technical report on methodology and experiments.
- `predictions.csv`: Generated labels for the competition test set.

## Quick Start

### 1. Installation
Clone the repository and install the necessary dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/SMILES-HALLUCINATION-DETECTION.git](https://github.com/YOUR_USERNAME/SMILES-HALLUCINATION-DETECTION.git)
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt


### 2. Running the Solution

To reproduce the extraction process, train the probe, and generate the `predictions.csv` file, execute:

```bash
python solution.py

```

## Dataset

The dataset consists of 689 labeled samples (`data/dataset.csv`) and a held-out test set (`data/test.csv`). Each sample includes a prompt in ChatML format, a model response, and a binary label.
