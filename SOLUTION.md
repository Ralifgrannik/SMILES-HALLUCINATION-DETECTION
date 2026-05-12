# Solution Report - SMILES-2026 Hallucination Detection

link to predictions.csv: https://drive.google.com/drive/folders/17zFKtMo73gRwW93uVp_Wa4XwslijE_tG?usp=sharing

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




### Technical Requirements:

* Python 3.9+
* CUDA-compatible GPU



## Solution Description

### Methodology

The proposed solution implements a **Linear Probing** architecture integrated with a robust feature selection mechanism. This approach addresses the high dimensionality of LLM hidden states relative to the limited sample size ($N=689$) through the following technical components:

1.  **Strategic Layer Selection & Aggregation (`aggregation.py`)**:
    * Layers **[14, 16, 18, 22]** were identified as optimal. This configuration captures the progression from semantic conceptualization (middle layers) to final logical convergence (late layers), while bypassing the noise inherent in terminal embedding transformations.
    * **Multimodal Pooling**: Both Mean and Max pooling operations were applied to each selected layer. The integration of **Max pooling** proved essential for detecting "activation spikes"—a known physiological heuristic for model uncertainty and potential hallucination.

2.  **Regularization and Classification (`probe.py`)**:
    * To mitigate the risk of overfitting within a high-dimensional feature space ($D=8064$), a Logistic Regression model with an **L1 (Lasso) penalty** was utilized ($C=0.05$).
    * This methodology facilitates intrinsic feature selection, effectively condensing the active feature set to approximately **65–102 salient neurons** that exhibit the strongest predictive correlation with factual accuracy.

3.  **Statistical Robustness via K-Fold (`splitting.py`)**:
    * A **5-fold Stratified Cross-Validation** strategy was implemented to ensure that the primary metric (AUROC) remains invariant to specific data partitions. This provides a generalized and reliable estimation of the probe's performance on the held-out competition test set.

### Key Performance Drivers

* **Sparsity Induction**: The transition from L2 to L1 regularization was the most significant factor in reducing the generalization gap between training and validation AUROC.
* **Dynamic Residuals (Residual Drift)**: Incorporating the "delta" (the activation variance between subsequent layers) enabled the probe to capture the model's "stochastic hesitation" during the autoregressive generation process.
