link to predictions.csv: https://drive.google.com/drive/folders/17zFKtMo73gRwW93uVp_Wa4XwslijE_tG?usp=sharing

# SMILES-2026 Hallucination Detection

This repository contains a binary classifier (probe) designed to detect hallucinations in the Qwen2.5-0.5B language model using its internal latent states.

## Project Overview
The aim of this project is to distinguish between truthful (label 0) and hallucinatory (label 1) model responses. By extracting and analyzing hidden representations from the transformer layers, we identify patterns that correlate with model uncertainty and falsification of facts.

## Results
- **Primary Metric (Test AUROC):** 73.68%

## Quick Start
### 1. Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ralifgrannik/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
```

If the data folder has not been downloaded:
```bash
cd SMILES-HALLUCINATION-DETECTION

# Create the data folder
mkdir -p data

# Download the data 
wget -O data/dataset.csv https://github.com/ahdr3w/SMILES-2026-Hallucination-Detection/raw/main/data/dataset.csv
wget -O data/test.csv https://github.com/ahdr3w/SMILES-2026-Hallucination-Detection/raw/main/data/test.csv
```

### 2. Running the Solution

To reproduce the extraction process, train the probe, and generate the `predictions.csv` file, execute:

```bash
python solution.py
```
