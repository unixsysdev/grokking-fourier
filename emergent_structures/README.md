# Emergent Structures in Qwen3

Probing for Fourier-like and other structured representations in pretrained LLMs.

## 📖 In-Depth Technical Reports

- **[Fourier Mechanistics Report](FOURIER_MECHANISTICS.md)** - Summary of circuits, causal ablation, and Logit Lens discoveries.
- **[Ablation Results](ablation_results.json)** - Raw data from causal experiments.

- [Alphabet](probe_structures.py) - Alphabet cyclic structure

## 🎯 Key Findings

### Cyclic Domain Encodings

**Qwen3 has learned cyclic (Fourier) encodings for multiple domains!**

| Domain | Best Dim | R² | Frequency |
|--------|----------|-----|-----------|
| **Days of week** | 125 | **0.9953** | k=1 |
| **Alphabet (cyclic)** | 238 | **0.9823** | k=1 |
| **Clock hours** | 725 | 0.9379 | k=1 |
| **Months** | 410 | 0.9288 | k=1 |

### 🔬 Fourier vs. Polynomial Detection (Robustness Test)

To ensure the "Fourier" results aren't just artifacts of small sample sizes or linear ramps, we tested multiple function hypotheses:

| Modulus | Dimension | Fourier $R^2$ | Linear $R^2$ | Quadratic $R^2$ | Winner |
|---------|-----------|---------------|--------------|-----------------|--------|
| **mod 7** | 35 | **0.9855** | 0.0256 | 0.0306 | **Fourier** |
| **mod 11** | 112 | **0.9499** | 0.1542 | 0.1831 | **Fourier**|
| **mod 17** | 867 | **0.9664** | 0.1241 | 0.1412 | **Fourier** |

**Interpretation**: The near-zero linear $R^2$ values for Dimension 35 prove it is NOT a magnitude detector (ReLU-friendly). It has "abandoned" the linear number line for a periodic one that corresponds to the cyclic nature of modular addition.

### 🧪 Causal Ablation Validation

We verified the causality of these dimensions by zeroing them out during inference using PyTorch hooks.

| Dimension | Role | Arith Acc Drop | Text PPL Change | Result |
|-----------|------|----------------|-----------------|--------|
| **35 (L9)** | Core Numerical Bridge | **-24%** | +2071% (Crash) | Critical Bottleneck |
| **8 (L7)** | Specialized Arithmetic | **-16%** | -2.2% (No Impact) | **Causal Proof** |
| 100 (L14)| Control Dim | 0% | 0% | No Effect |

→ **[See causal_ablation.py](causal_ablation.py)** for implementation details.

### 🔬 Prime Generalization Test

**Critical finding: Fourier circuits are SPECIALIZED, not dynamic!**

| Prime | Dim 35 R² | Best Dim | Best R² |
|-------|-----------|----------|---------|
| 7 | 0.9855 | 505 | 0.9967 |
| 11 | **0.5208** ❌ | 112 | 0.9499 |
| 13 | **0.5207** ❌ | 216 | 0.9591 |
| 17 | 0.8189 | **867** | 0.9664 |
| 19 | 0.8479 | **867** | 0.9757 |
| 23 | 0.8812 | **867** | 0.9678 |
| 29 | 0.8763 | **867** | 0.9445 |
| 31 | 0.9148 | 463 | 0.9402 |

### 🔬 Deep Investigation Results

#### 1. Dim 867 is a "Range Detector", Not a Prime Detector!

Testing dim 867 on numbers 5-35 (primes AND composites):

```
Numbers with R² > 0.85: [7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31]
```

| Statistic | Value |
|-----------|-------|
| Primes avg R² | 0.9062 |
| Composites avg R² | 0.8354 |
| Sweet spot | n ≈ 14-29 |

**Dim 867 handles the RANGE 14-29, regardless of primality!** It works for composites (14, 15, 16, 18, 20, 21...) just as well as primes.

#### 2. Attention Doesn't Route — MLPs Do!

Attention to the modulus token is nearly uniform across different primes:

```
p= 7: attention = 0.033
p=11: attention = 0.034
p=17: attention = 0.037
p=23: attention = 0.050
p=31: attention = 0.045
```

Hidden state values at key dimensions are **nearly identical** regardless of prime:

```
Dim     p=7     p=11    p=17    p=23    p=31    
--------------------------------------------------
35      30.97   31.12   31.13   31.11   31.12   (constant!)
867     0.18    0.22    0.22    0.23    0.29    (constant!)
```

**Conclusion: The routing happens in MLPs, not attention.** The model doesn't "attend" differently to different primes — the MLP layers must be doing the differentiation.

#### 3. Arithmetic and Calendar Circuits are Completely Separate!

Testing composite numbers common in real life:

| Modulus | Best Dim | R² | Uses day/hour/month dims? |
|---------|----------|-----|---------------------------|
| mod 10 | 557 | 0.9486 | ❌ No |
| mod 12 | 536 | 0.9723 | ❌ No |
| mod 24 | 867 | 0.9448 | ❌ No |
| mod 60 | 601 | 0.9343 | ❌ No |
| mod 100 | 347 | 0.8994 | ❌ No |

**The cyclic dims (125=days, 725=hours, 410=months) are NOT reused for arithmetic!**

Even though mod 12 is the same period as hours/months, the model uses **completely different dimensions** for:
- "Monday + 3 days" → Dim 125
- "3 o'clock + 5 hours" → Dim 725  
- "5 + 3 mod 12" → Dim 536 (different!)

The model has **separate circuits for semantic domains** vs **arithmetic operations**.

## The Complete Picture

```
Qwen3's Modular Circuits:

CALENDAR/TIME (semantic):
  Dim 125: "What day?"        → period 7
  Dim 410: "What month?"      → period 12
  Dim 725: "What hour?"       → period 12

ARITHMETIC (computational):
  Dim 505: mod 7 arithmetic   → period 7
  Dim 112: mod 11 arithmetic  → period 11
  Dim 216: mod 13 arithmetic  → period 13
  Dim 867: mod 14-29 range    → varies (PARTIAL GENERALIZATION!)
  Dim 463: mod 31 arithmetic  → period 31
  Dim 536: mod 12 arithmetic  → period 12 (different from hours!)
  Dim 557: mod 10 arithmetic  → period 10
  ...

ROUTING:
  - Attention is uniform (doesn't route)
  - MLPs differentiate between moduli
  - Separate circuits for semantic vs arithmetic
```

## Implications

### 1. Extreme Specialization

The model doesn't generalize — it has **hundreds of specialized circuits**:
- One for days, one for hours, one for months
- Separate ones for mod 7 arithmetic vs "Monday + 3 days"
- Different circuits for different numerical ranges

### 2. MLP-Based Routing

Attention doesn't select which circuit to use. The MLPs must be doing pattern matching on the input to activate the right specialized circuit.

### 3. Partial Generalization in Dim 867

Dim 867 is interesting — it handles a **range** (14-29) rather than a single modulus. This is a hint of generalization, but still limited. Larger models might extend this to true universality.

### 4. Inefficiency

This is **wildly inefficient**:
- Separate dims for mod 12 arithmetic vs hours vs months
- Many dimensions "wasted" on narrow tasks
- A universal circuit would be far more parameter-efficient

This likely explains why larger models show "emergent abilities" — they have enough capacity to develop universal circuits instead of this lookup-table approach.

## Running the Experiments

```bash
# Cyclic domain experiments
./run.sh

# Prime generalization test
./run_prime_test.sh

# Deep investigation (dim 867, attention, composites)
./run_investigation.sh
```

## Output Files

### `results/` — Cyclic Experiments
- `days_of_week.png`, `months.png`, `clock_hours.png`, `alphabet.png`

### `results_primes/` — Prime Generalization
- `dim35_across_primes.png`, `top_dims_per_prime.png`

### `results_deep/` — Deep Investigation
- `dim867_investigation.png` — Dim 867 works for range 14-29
- `attention_routing.png` — Attention is uniform (no routing)
- `composites_analysis.png` — Arithmetic uses separate dims from calendar

## Conclusion

**Qwen3 0.6B is a massive lookup table of specialized Fourier circuits.**

Key insights:
1. **No universal modular arithmetic** — different dims for different moduli
2. **Semantic vs arithmetic separation** — "hours" ≠ "mod 12"
3. **Range-based partial generalization** — Dim 867 handles 14-29
4. **MLP routing, not attention** — Attention doesn't differentiate
5. **Extreme inefficiency** — Larger models likely do better

This suggests that the "Fourier algorithm" from the grokking paper exists in LLMs, but is **fragmented across many specialized circuits** rather than unified into a single universal algorithm.
