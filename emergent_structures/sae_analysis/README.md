# Sparse Autoencoder (SAE) Analysis of Layer 7

This folder contains the implementation and results for decomposing Qwen3's hidden representations into sparse features.

## 🎯 Goal
To investigate **Superposition**: Is a single dimension (like Dimension 8) doing multiple things? By projecting the 1024-dimensional space into 4096 sparse features, we can find "pure" features that correspond strictly to modular arithmetic.

## 📂 Contents
- `train_sae.py`: Trains a Sparse Autoencoder on Layer 7 activations from a mix of math and text.
- `analyze_sae_features.py`: Scans the learned features for Fourier structure using $R^2$ fitting.

## 🚀 How to Run
```bash
# 1. Train the SAE on Layer 7
python train_sae.py

# 2. Analyze the sparse features for Fourier signals
python analyze_sae_features.py
```

## 📈 Expected Results
We expect to find a small subset of the 4096 features that exhibit near-perfect sine waves ($R^2 > 0.99$), potentially cleaner than the raw neurons. This would prove that the model uses specific "directions" in activation space for math, even if they share dimensions with language features.
