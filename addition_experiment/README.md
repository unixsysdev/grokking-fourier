# Addition Experiment

Teaching a transformer to learn addition from scratch using character-level representation.

## The Challenge

Can a small transformer actually **learn the addition algorithm** rather than just memorizing input-output pairs?

Key insight: If the number space is large enough (5-digit numbers = 10 billion possible pairs) and we train on only a tiny fraction (~50k samples), the model **must** learn the algorithm to generalize.

## Architecture

```
Input:  "12345+67890=" (character tokens)
Output: "80235" (generated digit-by-digit)

Model: Encoder-Decoder Transformer
- Encoder: 3 layers, processes the equation
- Decoder: 3 layers, generates answer autoregressively
- d_model: 128, n_heads: 4, d_ff: 512
- ~1.5M parameters
```

## Training Strategy

### Curriculum Learning

| Phase | Epochs | Numbers | What Model Learns |
|-------|--------|---------|-------------------|
| 1 | 0-5k | 1-2 digits | Basic single-digit addition |
| 2 | 5k-15k | up to 3 digits | Two-digit addition, simple carries |
| 3 | 15k-30k | up to 4 digits | Multi-digit addition |
| 4 | 30k+ | up to 5 digits | Full complexity |

### Data

- **Training**: 50,000 random pairs (0.0000005% of possible 5-digit pairs!)
- **Test Interpolation**: Different pairs in same 1-5 digit range
- **Test Extrapolation**: 6-digit numbers (never seen in training)

## Quick Start

```bash
cd addition_experiment

# Start training
python train.py --n_epochs 50000

# Resume if interrupted
python train.py --resume

# Check accuracy
python sweep_accuracy.py --epoch latest

# Generate analysis plots
python generate_compendium.py latest
```

## Files

| File | Description |
|------|-------------|
| `model.py` | AdditionTransformer architecture |
| `train.py` | Training loop with curriculum learning |
| `sweep_accuracy.py` | Test accuracy across categories |
| `generate_compendium.py` | Generate analysis visualizations |

## What Success Looks Like

| Metric | Target | Meaning |
|--------|--------|---------|
| Train Accuracy | 99%+ | Fits training data |
| Interpolation | 95%+ | **Learned the algorithm!** |
| Extrapolation (6-digit) | 80%+ | **Truly generalizes!** |

## Analysis Outputs

After running `generate_compendium.py`:

```
analysis/compendium_e{epoch}/
├── training_history.png        # Loss and accuracy curves
├── attention_patterns.png      # What does decoder attend to?
├── carry_analysis.png          # Accuracy on carry-heavy problems
├── digit_embeddings.png        # Are digits meaningfully organized?
├── position_accuracy.png       # Which digit positions are hardest?
├── fourier_embeddings.png      # Circular structure in digit embeddings?
├── weight_fft.png              # Frequency analysis of weight matrices
├── neuron_activations.png      # Which neurons detect what patterns?
├── periodic_neurons.png        # Sinusoidal activation patterns (Fourier!)
├── periodic_neurons_detail.png # Detailed view of top periodic neurons
├── output_logits.png           # Output probability distributions
└── grokking_progress.png       # Has the model grokked yet?
```

### Fourier Analysis

The compendium includes analysis inspired by the "grokking" literature on modular arithmetic. We look for:

- **Circular digit embeddings**: Do digits 0-9 form a circle in embedding space?
- **Sinusoidal neuron activations**: Do neurons fire in periodic patterns based on (a+b)?
- **Fourier features**: Are there frequency peaks in the weight matrices?

If the model learns a Fourier-based algorithm for addition (similar to how transformers learn modular arithmetic), we'd expect to see clear sinusoidal patterns in neuron activations.

**Quick Fourier analysis only:**
```bash
python generate_compendium.py --periodic-only
```

```

## Key Questions This Experiment Answers

1. **Does the model learn positional value?** (that "1" in position 3 means 100)
2. **Does it learn carrying?** (9+1=10, carry the 1)
3. **Does it process right-to-left?** (like humans do addition)
4. **Can it extrapolate to larger numbers?**

## Why Character-Level?

If we used token-per-number (e.g., token_42, token_37), the model could just memorize:
```
token_42 + token_37 = token_79
```

With character-level, even if it memorizes `4+3=7` and `2+7=9`, it still needs to understand:
- Positional alignment
- Carrying between positions  
- Variable-length outputs

This forces learning the actual algorithm!
