
# Solution Report - SMILES-2026 Hallucination Detection

## Reproducibility
This repository contains a self-contained pipeline for detecting hallucinations in the Qwen2.5-0.5B model. 
The solution is optimized for execution in a CUDA-enabled environment (e.g., Google Colab T4 GPU).

### Execution Instructions:
1. **Environment Setup**:
   Install the required dependencies as specified in the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Integrity**:
Ensure that `dataset.csv` and `test.csv` are present in the `./data/` directory.
3. **Inference and Training**:
Run the primary execution script to perform feature extraction, probe training, and prediction generation:
```bash
python solution.py
```



### Technical Requirements:

* Python 3.9+
* CUDA-compatible GPU
* Execution time: Approximately 185 seconds for full feature extraction on 689 samples.

---

## Final Solution Description

### Methodology

The proposed solution implements a **Linear Probing** architecture integrated with a robust feature selection mechanism. The approach addresses the high dimensionality of LLM hidden states relative to the limited sample size ($N=689$) through the following components:

1. **Strategic Layer Selection (`aggregation.py`)**:
* Layers **[14, 16, 18, 22]** were identified as optimal. This configuration captures the transition from semantic conceptualization (mid-layers) to final logical convergence (late layers), avoiding the noise present in the terminal embedding transformation.
* **Multi-modal Pooling**: Both Mean and Max pooling were applied to each layer. Max pooling, in particular, proved essential for identifying "activation spikes" — a known heuristic for model uncertainty and potential hallucination.


2. **L1-Regularized Linear Classifier (`probe.py`)**:
* To mitigate overfitting in a high-dimensional feature space ($D=8064$), a **Logistic Regression model with L1 (Lasso) penalty** was utilized ($C=0.05$).
* This approach performed intrinsic feature selection, reducing the active feature set to approximately **65–102 salient neurons**.


3. **Statistical Robustness via K-Fold (`splitting.py`)**:
* A **5-fold Stratified Cross-Validation** strategy was implemented. This ensured that the primary metric (AUROC) is invariant to specific data splits and provided a more generalized estimation of the probe's performance on the held-out test set.



### Key Performance Drivers

* **Sparsity induction**: The transition from L2 to L1 regularization was the most significant factor in closing the gap between training and test AUROC.
* **Dynamic Residuals**: Incorporating the "delta" (residual drift) between the final layers helped capture the model's "hesitation" during token generation.

---

## Experiments and Failed Attempts

1. **Principal Component Analysis (PCA)**:
* *Result*: Discarded.
* *Analysis*: While PCA reduced dimensionality, it projected features into a non-interpretable space that diluted the signal of individual "truth-telling" neurons. Lasso proved superior by preserving the physical meaning of neural activations.


2. **Full Sequence Context**:
* *Result*: Sub-optimal.
* *Analysis*: Including the prompt embeddings introduced significant noise. The signal for hallucination is most concentrated in the final **20-25 tokens** of the response, where the model's autoregressive state accumulation is most pronounced.


3. **Global Average Pooling (GAP)**:
* *Result*: Replaced with Mean+Max pooling.
* *Analysis*: GAP alone was insufficient to capture the extreme activation values that characterize "confused" states in small-scale models like Qwen2.5-0.5B.
