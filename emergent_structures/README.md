# Emergent Structures in Qwen3

Probing for Fourier-like and other structured representations in pretrained LLMs.

## 🎯 Key Findings

### Cyclic Domain Encodings

**Qwen3 has learned cyclic (Fourier) encodings for multiple domains!**

| Domain | Best Dim | R² | Frequency | Notes |
|--------|----------|-----|-----------|-------|
| **Days of week** | 125 | **0.9953** | k=1 | Near-perfect! |
| **Alphabet (cyclic)** | 238 | **0.9823** | k=1 | Near-perfect! |
| **Clock hours** | 725 | 0.9379 | k=1 | Strong |
| **Months** | 410 | 0.9288 | k=1 | Strong |
| **Alphabet (linear)** | 458 | 0.8907 | - | Linear position |
| **Analogies** | - | 0.92 cos | rank ~24 | Parallelogram exists but noisy |

### 🔬 Prime Generalization Test

**Critical finding: The Fourier dimensions are SPECIALIZED, not dynamic!**

We tested whether dimension 35 (which showed R²=0.96 for p=23) generalizes to other primes:

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

**Verdict: SPECIALIZED, not dynamic**

- Dim 35 fails for p=11,13 (R² ≈ 0.52)
- **Dim 867** dominates for medium primes (17-29)
- Different small primes have different best dimensions
- The model has a **lookup table of specialized circuits**, not a general-purpose modular arithmetic unit

### What This Means

```
NOT this (dynamic):
  Input: "a + b mod p" → [Universal Fourier Circuit] → cos(2π(a+b)/p)

BUT this (specialized):
  Input: "a + b mod 7"  → Dim 505 circuit → cos(2π(a+b)/7)
  Input: "a + b mod 11" → Dim 112 circuit → cos(2π(a+b)/11)
  Input: "a + b mod 17" → Dim 867 circuit → cos(2π(a+b)/17)
  ...
```

The model learned **separate Fourier circuits for different prime ranges**, similar to how it has separate circuits for days (dim 125) vs months (dim 410).

## Interpretation

### Different Dimensions for Different Domains

The model has **dedicated dimensions** for each cyclic domain:

```
Dim 125: "What day is it?"         → cosine wave (period 7)
Dim 238: "What letter position?"   → cosine wave (period 26)
Dim 410: "What month?"             → cosine wave (period 12)
Dim 725: "What hour?"              → cosine wave (period 12)

Dim 505: "mod 7 arithmetic"        → cosine wave (period 7)
Dim 112: "mod 11 arithmetic"       → cosine wave (period 11)
Dim 867: "mod 17-29 arithmetic"    → cosine wave (varies)
Dim 463: "mod 31 arithmetic"       → cosine wave (period 31)
```

### Why Specialized Rather Than Dynamic?

1. **Training distribution**: The model saw specific primes more often (7, 10, 12 from time/calendar, small primes from math)
2. **No pressure to generalize**: Each prime is a separate "task" in training data
3. **Easier to memorize**: Specialized circuits are simpler than a universal algorithm
4. **Attention routing**: The model probably routes different primes to different circuits based on the value of p in the prompt

### The Dim 867 Mystery

Dimension 867 is interesting — it handles primes 17, 19, 23, 29 (but not 7, 11, 13, 31). This suggests:
- It may have learned a "medium prime" circuit
- Or these primes share some property that activates the same pathway
- Worth investigating further!

## Experiments

### 1. Days of the Week (Period 7)
**Result**: R² = 0.9953 in dimension 125

### 2. Months of the Year (Period 12)
**Result**: R² = 0.9288 in dimension 410

### 3. Clock Hours (Period 12)
**Result**: R² = 0.9379 in dimension 725

### 4. Alphabet Position (Period 26)
**Result**: R² = 0.9823 (cyclic), R² = 0.8907 (linear)

### 5. Word Analogies
**Result**: Cosine similarity 0.92, average rank ~24

### 6. Prime Generalization Test
**Result**: Specialized circuits, not dynamic computation

## Running

```bash
# Cyclic domain experiments
./run.sh

# Prime generalization test
./run_prime_test.sh
```

## Output Files

### Cyclic Experiments (`results/`)
- `days_of_week.png`, `months.png`, `clock_hours.png`, `alphabet.png`
- `analogies.png`, `results.json`

### Prime Generalization (`results_primes/`)
- `dim35_across_primes.png` — Dim 35 cosine fits for each prime
- `dim35_r2_by_prime.png` — Bar chart showing R² drops for some primes
- `top_dims_per_prime.png` — Heatmap of which dims work for which primes
- `results.json` — Full numerical results

## Conclusion

**Fourier representations are fundamental, but specialized rather than universal.**

The model learned:
1. **Separate "clocks"** for days, months, hours, alphabet
2. **Separate arithmetic circuits** for different prime ranges
3. **No general-purpose modular arithmetic** — it's more like a lookup table

This suggests LLMs develop **task-specific computational primitives** rather than universal algorithms. The Fourier structure is real, but it's instantiated separately for each domain the model encountered during training.

### Future Questions

1. **How does attention route** different primes to different dimensions?
2. **Is dim 867 special?** Why does it handle primes 17-29?
3. **Can fine-tuning unify** the specialized circuits into a universal one?
4. **Do larger models** (7B, 70B) have more universal circuits?
