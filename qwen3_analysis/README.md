# Qwen3 0.6B Fourier Analysis

**Question**: Does a pretrained LLM (Qwen3 0.6B) show similar Fourier structure to small transformers trained from scratch on modular arithmetic?

## 🎯 Key Finding

**YES!** Pretrained Qwen3 0.6B shows **emergent Fourier-like structure** for modular arithmetic, despite never being specifically trained on this task.

| Dimension | Best R² | Best Layer | Frequency |
|-----------|---------|------------|-----------|
| **35** | **0.9578** | Layer 9 | k=1 |
| 8 | 0.9178 | Layer 7 | k=1 |
| 1 | 0.8933 | Layer 1 | k=1 |
| 7 | 0.8784 | Layer 2 | k=1 |
| 243 | 0.8536 | Layer 4 | k=1 |

**Dimension 35 achieves R² = 0.9578** — meaning **95.8% of its variance** is explained by a single cosine function `cos(2π(a+b)/p)`!

## Background

The grokking paper (Nanda et al., 2023) found that small transformers trained on `(a + b) mod p` learn a beautiful algorithm:
- They embed inputs as `sin(wₖa), cos(wₖa)` at sparse "key frequencies"
- They use trigonometric identities to compute `cos(wₖ(a+b))`
- The network's weights become **sparse in the Fourier basis**

**This experiment**: We prompt Qwen3 0.6B with modular arithmetic problems and analyze its internal activations for similar structure.

## Method

1. Create prompts: `"a + b mod p = "` for all pairs `(a, b)` where `a, b ∈ [0, p-1]`
2. Run inference and extract hidden states from various layers
3. Apply 2D Fourier transform to the activation grid
4. Analyze how well individual dimensions fit `cos(2πk(a+b)/p)` for various frequencies k

### Key Insight: Sum-Organization

For each dimension, we reorganize activations by the sum `s = (a+b) mod p`:
- Group all (a,b) pairs that produce the same sum s
- Average activations within each group
- This gives us a 1D signal indexed by s ∈ [0, p-1]
- Fit cosine functions to this signal

If the dimension encodes Fourier information, this 1D signal should be nearly sinusoidal!

## Results

### Initial Layer Scan

We first scanned layers 3, 7, 14, 21, 27 to find which dimensions appeared most "Fourier-like":

| Layer | Significant 2D Freqs | Gini Coeff | Top Dimensions |
|-------|---------------------|------------|----------------|
| 3 | 13 | 0.9697 | 35, 8, 7, 1, 0 |
| 7 | 11 | 0.9704 | 35, 8, 7, 1, 243 |
| 14 | 11 | 0.9704 | 35, 8, 7, 1, 243 |
| 21 | 11 | 0.9704 | 35, 8, 7, 1, 243 |
| 27 | 11 | 0.9703 | 35, 8, 7, 1, 243 |

**Dimension 35 appears in ALL layers** as the top Fourier dimension!

### Deep Dive Analysis

We then analyzed dimensions 35, 8, 7, 1, and 243 across **all 28 layers**:

**Dimension 35** shows the strongest Fourier structure:
- R² peaks at **0.9578** in layer 9
- Encodes frequency k=1 (the fundamental frequency)
- Structure is visible from layer 0 and strengthens through middle layers

**All dimensions encode k=1**: The fundamental frequency `cos(2π(a+b)/p)` dominates, rather than higher harmonics.

### What the Plots Show

**`dim35_sum_activations.png`**: For each layer, shows activation vs (a+b) mod p. In layer 9, this is nearly a perfect cosine wave!

**`dim35_metrics_across_layers.png`**: Shows how R², Gini coefficient, and dominant frequency evolve across layers.

**`dim35_fft_heatmap.png`**: FFT magnitude heatmap showing frequency k=1 dominates across all layers.

**`all_dims_comparison.png`**: Compares R² across all analyzed dimensions.

## Interpretation

### What This Means

1. **Pretrained LLMs develop numerical representations**: Even without explicit arithmetic training, Qwen3 has dimensions that naturally encode periodic functions of sums.

2. **Similar to grokking, but emergent**: The small model in the grokking paper was *forced* to learn Fourier structure. Qwen3 developed it as a byproduct of general language modeling.

3. **Structure is localized**: Not all dimensions are Fourier-like. Dimension 35 is special — it carries most of the periodic sum information.

4. **Early-middle layers are key**: The strongest Fourier fits occur in layers 1-9, suggesting basic numerical processing happens early in the network.

### Caveats

- The model's actual accuracy on modular arithmetic is only ~28% — it has the representation but not perfect computation
- R² values (0.85-0.96) are high but not perfect, suggesting other computations overlap
- We only tested p=23; larger primes might show different structure

## Running the Experiments

### Quick Start

```bash
# Basic layer scan
./run.sh

# Deep analysis of specific dimensions
./run_deep.sh
```

### Manual Usage

```bash
source ../venv/bin/activate

# Layer scan (p=23, ~529 pairs)
python analyze_qwen3.py --p 23 --layers "3,7,14,21,27"

# With accuracy testing
python analyze_qwen3.py --p 23 --layers "7,14,21" --test_accuracy

# Deep analysis of specific dimensions
python analyze_deep.py --p 23 --dims "35,8,7,1,243"

# Find top Fourier dimensions in a specific layer
python analyze_deep.py --p 23 --find_dims --find_layer 14
```

## Output Files

### Basic Analysis (`results/`)

| File | Description |
|------|-------------|
| `summary.json` | Numerical results for all layers |
| `fourier_analysis_layer*_hidden.png` | 2D FFT magnitude and significant frequencies |
| `top_dims_layer*_hidden.png` | Activation patterns for most Fourier-like dimensions |

### Deep Analysis (`results_detailed/`)

| File | Description |
|------|-------------|
| `detailed_analysis.json` | Full numerical results for all dimensions across all layers |
| `dim*_activation_patterns.png` | 2D activation heatmaps (a,b) across all 28 layers |
| `dim*_sum_activations.png` | Activation vs (a+b) mod p for each layer (shows the cosine!) |
| `dim*_metrics_across_layers.png` | R², Gini, frequency evolution across layers |
| `dim*_fft_heatmap.png` | FFT magnitude heatmap across layers |
| `all_dims_comparison.png` | Compare R² for all analyzed dimensions |

## Comparison to Grokking Paper

| Aspect | Grokking Paper | This Experiment |
|--------|---------------|-----------------|
| Model size | ~226k params | 600M params |
| Training | From scratch on mod-add | Pretrained on text |
| Task specificity | Single task | General purpose |
| Layers | 1 layer | 28 layers |
| Best R² for cos fit | ~1.0 (designed for it) | **0.9578** (emergent!) |
| Key frequencies | Multiple (4, 11, 14, 26, 35) | Primarily k=1 |

The fact that a general-purpose LLM shows *any* Fourier structure is remarkable. It's not as clean as the purpose-built model, but it's far from random!

## Conclusion

**Pretrained LLMs spontaneously develop Fourier-like representations for arithmetic.** Dimension 35 in Qwen3 0.6B encodes `cos(2π(a+b)/p)` with R² > 0.95 — strong evidence that the "Fourier computation hypothesis" from the grokking paper extends to general language models.

This suggests a deeper truth: **neural networks may naturally discover periodic representations for modular arithmetic**, whether trained from scratch or through general language modeling.

## References

- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability", ICLR 2023
- Qwen3: https://huggingface.co/Qwen/Qwen3-0.6B
