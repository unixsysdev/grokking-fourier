# Mechanistic Analysis of Fourier Circuits in Qwen3

This document summarizes our deep-dive investigation into how Qwen3-0.6B represents and computes modular arithmetic through emergent Fourier oscillators.

## 🎯 Executive Summary
We discovered that Qwen3 uses a **specialized trigonometric algorithm** to solve modular addition. This algorithm is localized to a small number of high-variance dimensions in the first third of the network (Layers 1-14). 

## 🏗️ The Identified Circuits

| Dimension | Peak Layer | $R^2$ (Fourier) | Functional Role |
|-----------|------------|-----------------|-----------------|
| **35** | **9** | **0.9578** | **Fundamental Clock ($k=1$)**: The primary mapping of numbers to a periodic cycle. |
| **8** | **7** | **0.9178** | **Modular Arithmetic Unit**: Specialized neuron for addition; ablate-safe for general text. |
| **867** | **14** | **0.9664** | **Range Specialist**: Handles larger moduli and provides additive specialized logic. |
| **1** | **1** | **0.8933** | **Early Projection**: Immediate mapping of tokens into cyclic space. |

## 🔬 Key Experiments & Findings

### 1. Fourier vs. Polynomial (Robustness)
**Question**: Is the model just "counting" (Linear ramp) or truly "oscillating" (Fourier)?
**Result**: Near-zero $R^2$ for linear/quadratic fits for Dim 35. The model has completely abandoned linear magnitude logic for trigonometric logic in these dimensions.

### 2. Causal Ablation (Necessity)
**Question**: Are these dimensions actually required for math?
**Result**: 
- Ablating **Dimension 8 (L7)** drops math accuracy by **16%** but has **0% impact** on general text. This proves it is a dedicated "math neuron."
- Ablating **Dimension 35 (L9)** drops math accuracy to **4%** but crashes general grammar, showing it is a foundational numerical bottleneck.

### 3. Logit Lens (Internal Thought Process)
**Question**: When does the model "know" the answer?
**Result**: The Fourier calculation happens early (Layers 7-14). The Logit Lens shows the probability of the correct answer token doesn't "peak" in the output vocabulary until **Layer 25**. 
- *Finding*: In some failed generations, the Logit Lens showed the model had the **correct answer internally** at Layer 25, but lost it in the final cleanup layers.

## 📂 Experimental Artifacts

- [`detect_polynomial_functions.py`](detect_polynomial_functions.py): Robustness testing vs linear/cubic hypotheses.
- [`causal_ablation.py`](causal_ablation.py): Hook implementation for zeroing out dimensions.
- [`logit_lens_analysis.py`](logit_lens_analysis.py): Projecting hidden states to vocabulary.
- [`ablation_results.json`](ablation_results.json): Raw data from causal tests.

## 🖼️ Visualizations
*(Generated during analysis)*
- `logit_lens_results/`: Heatmaps showing probability evolution across 28 layers.
- `polynomial_test/`: Comparison plots of Fourier vs. Polynomial fits.

### 4. Sparse Feature Decomposition (Superposition Test)
**Question**: Is Dimension 8 doing two things at once?
**Method**: Trained a Sparse Autoencoder (SAE) on Layer 7 with 4096 features.
**Result**: **Feature 2822** emerged as a nearly-perfect Fourier oscillator ($R^2 = 0.9717, k=1$). 
- *Discovery*: While the raw Dimension 8 has an $R^2$ of ~0.91, the SAE successfully extracted a **purer direction in activation space** that scores significantly higher. 
- *The "Power" Feature*: **Feature 1834** showed a lower $R^2$ (0.83) but an extremely high activation magnitude (20.4). This suggests it might be the "powerhouse" that drives the math signal, while Feature 2822 provides the "precision" phase.

→ **[See sae_analysis/README.md](sae_analysis/README.md)** for detailed feature plots.

---
**Conclusion**: Qwen3 acts as a "Lookup Table of Fourier Circuits." It has not learned a universal modular arithmetic algorithm but has instead evolved multiple specialized trigonometric units for different numerical ranges.
