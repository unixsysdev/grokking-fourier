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

## Phase 2: Results & The "Grokking Leap" (In-Progress)
As of December 21st, 07:55 AM, we have observed a major breakthrough using the **SinPE (Sinusoidal Modulus Encoding)** architecture.

### 1. The Breakthrough (Epoch 40,000)
- **Supervised CE**: The model has officially begun the "Grokking Leap."
  - **Mastery**: 100% accuracy on seen primes (e.g., mod 13).
  - **Generalization**: 67.5% accuracy on unseen large primes (e.g., mod 71).
  - **Fourier Purity**: Neurons have reached **0.74 SNR**, indicating very clean internal circular signals.
- **Cliff Mitigation**: Accuracy on extrapolation primes (mod 101) has increased from **0% to ~11%**, proving that the SinPE architecture is successfully transmitting a universal mathematical signal.

### 2. RL Progress
- The Policy Gradient model is taking longer to converge but shows a higher baseline stability for small primes. It is currently in the "Exploration Phase" with rising rewards.

### 3. Key Files
- `train_ce.py` / `train_rl.py`: The twin training tracks.
- `sweep_accuracy.py`: A new utility to measure accuracy across all primes [2, 113].
- `analyze_miras_mechanics.py`: Updated to support Phase 2 SinPE models.

## Usage
To evaluate the models:
```bash
# General analysis
python miras_experiment/analyze_miras_mechanics.py --mode ce

# Exhaustive accuracy sweep
python miras_experiment/sweep_accuracy.py ce 40000
```
