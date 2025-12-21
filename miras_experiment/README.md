# MIRAS Experiment: Universal Fourier Grokking

This directory contains the code and analysis for the MIRAS-augmented transformer experiment, aimed at achieving "Universal Grokking" of modular addition.

## Previous Findings (Phase 1)
- **Architecture**: Discrete Prime Embeddings + TitansMemory.
- **Success**: Model achieved ~97% accuracy on unseen primes within its training range (up to $p=79$).
- **Failure**: Encountered the "Extrapolation Cliff" at $p=83$, where accuracy dropped to <10%.
- **Hypothesis**: Discrete embeddings treat primes as categorical tokens rather than scalar magnitudes, preventing mathematical extrapolation to higher $p$.

## Phase 2: Infinite Extrapolation (Current)
We are now testing a new architecture designed for scale-invariant generalization.

### 1. Architectural Changes
- **Sinusoidal Modulus Encoding (SinPE)**: Replaced discrete prime embeddings with a continuous sinusoidal signal. This provides the model with the "scalar magnitude" information necessary for periodic alignment across all $p$.

### 2. Dual-Track Experiments
We are running two parallel training strategies to compare "Mimicry" vs "Discovery":
- **Cross-Entropy (`train_ce.py`)**: standard supervised learning.
- **Policy Gradient / RL (`train_rl.py`)**: REINFORCE algorithm to encourage the model to discover the most robust algorithmic strategy.

### 3. Usage
To start the experiments:
```bash
# Run Cross-Entropy SIN-PE
python miras_experiment/train_ce.py

# Run Policy Gradient RL SIN-PE
python miras_experiment/train_rl.py
```

## Goals
- Achieve 95%+ accuracy on primes far beyond the training distribution (up to $max\_p=150$ and potentially beyond).
- Verify "Sharper" Fourier signals in the RL-trained model.
- Conquer the "Seen Prime Performance Gap."
