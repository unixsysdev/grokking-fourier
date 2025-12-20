# Grokking Fourier Analysis

Replication of the Fourier analysis from:

**"Progress Measures for Grokking via Mechanistic Interpretability"**  
Nanda et al., ICLR 2023  
Paper: https://arxiv.org/abs/2301.05217

## What this does

Trains a small 1-layer transformer on modular addition (`a + b mod p`) and analyzes the learned algorithm using Fourier transforms.

The paper discovered that these networks learn a beautiful algorithm:
1. Embed inputs as `sin(wₖa), cos(wₖa)` at specific "key frequencies"
2. Use trigonometric identities to compute `cos(wₖ(a+b))`
3. Read off logits via `cos(wₖ(a+b-c))`, which peaks when `c = (a+b) mod p`

## Experimental Results

### Successful Grokking (p=113)

The model successfully grokked with the paper's original parameters:

| Metric | Value |
|--------|-------|
| Prime (p) | 113 |
| Train samples | 3,830 (30% of 12,769) |
| Test samples | 8,939 |
| Epochs | 25,000 |
| **Final test accuracy** | **99.9%** |
| Final test loss | 0.0345 |

**Key frequencies discovered: `[4, 11, 14, 26, 35]`**

The model learned a sparse Fourier representation with only 5 key frequencies (plus constant), exactly as the paper predicted. The 2D Fourier transform of the logits shows only 9 significant components.

### Failed Runs (p=53, p=71)

Smaller primes did not grok with weight_decay=1.0:
- p=53: Test accuracy stuck at ~1.3% after 15k epochs
- p=71: Test accuracy stuck at ~0.5% after 20k epochs

This matches the paper's finding that smaller primes require higher weight decay (λ=5.0) because the memorization solution is relatively cheaper.

## Quick Start

```bash
# Create virtual environment (first time only)
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib tqdm

# Run the full experiment
./run.sh
```

This will:
1. Train the model (~12 minutes on M3 Mac)
2. Run Fourier analysis
3. Generate plots in `analysis_p113/`

## Manual Usage

```bash
source venv/bin/activate

# Train with custom parameters
python train.py --p 113 --n_epochs 25000 --output_dir my_checkpoints

# Analyze a trained model
python analyze.py my_checkpoints/checkpoint_final.pt --output_dir my_analysis
```

## Key Parameters

- `p`: Prime modulus (113 recommended, matches paper)
- `train_frac`: Fraction of data for training (0.3 = 30%)
- `weight_decay`: Crucial for grokking! (1.0 for p≥113, 5.0 for smaller primes)
- `n_epochs`: Training epochs (grokking typically happens around 10k-15k)

## Output Files

After running, check the analysis folder for:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss and accuracy over training (shows grokking moment) |
| `embedding_fourier.png` | Fourier structure of embedding matrix W_E |
| `neuron_logit_fourier.png` | Fourier structure of neuron-logit map W_L |
| `neuron_activations.png` | Periodicity patterns in MLP neurons |
| `attention_patterns.png` | Periodicity in attention heads |
| `logits_2d_fourier.png` | 2D Fourier transform of output logits |

## Repository Structure

```
grokking-fourier/
├── model.py           # One-layer transformer architecture
├── train.py           # Training loop with AdamW + weight decay
├── analyze.py         # Fourier analysis and plotting
├── run.sh             # Quick run script
├── checkpoints_p113/  # Trained model (p=113, successful grokking)
├── analysis_p113/     # Fourier analysis plots
├── checkpoints_p71/   # Failed run (p=71)
├── checkpoints/       # Failed run (p=53)
└── README.md
```

## References

```bibtex
@inproceedings{nanda2023progress,
  title={Progress Measures for Grokking via Mechanistic Interpretability},
  author={Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
