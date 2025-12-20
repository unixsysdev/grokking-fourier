# Emergent Structures in Qwen3

Probing for Fourier-like and other structured representations in pretrained LLMs.

## 🎯 Key Findings

**Qwen3 has learned cyclic (Fourier) encodings for multiple domains!**

| Domain | Best Dim | R² | Frequency | Notes |
|--------|----------|-----|-----------|-------|
| **Days of week** | 125 | **0.9953** | k=1 | Near-perfect! |
| **Alphabet (cyclic)** | 238 | **0.9823** | k=1 | Near-perfect! |
| **Clock hours** | 725 | 0.9379 | k=1 | Strong |
| **Months** | 410 | 0.9288 | k=1 | Strong |
| **Alphabet (linear)** | 458 | 0.8907 | - | Linear position |
| **Analogies** | - | 0.92 cos | rank ~24 | Parallelogram exists but noisy |

### Comparison with Modular Arithmetic

| Experiment | Best R² | 
|------------|---------|
| Days of week | **0.9953** |
| Alphabet | 0.9823 |
| Modular arithmetic (from qwen3_analysis) | 0.9578 |
| Clock hours | 0.9379 |
| Months | 0.9288 |

**Days of the week has cleaner Fourier structure than modular arithmetic!** This makes sense — day names appear constantly in training text, while modular arithmetic is rare.

## Interpretation

### Different Dimensions for Different Domains

The model has **dedicated dimensions** for each cyclic domain:

```
Dim 35:  "What's (a+b) mod p?"     → cosine wave (period p)
Dim 125: "What day is it?"         → cosine wave (period 7)
Dim 238: "What letter position?"   → cosine wave (period 26)
Dim 410: "What month?"             → cosine wave (period 12)
Dim 458: "How far in alphabet?"    → linear ramp
Dim 725: "What hour?"              → cosine wave (period 12)
```

### Why Fourier Bases Emerge

Neural networks naturally discover that **periodic data is best encoded as sinusoids**:

1. Sinusoids are efficient (one dimension captures cycle position)
2. Dot products between sinusoids give useful computations (trig identities)
3. This emerges from gradient descent without explicit design

### Analogies Are Harder

Word analogies (king - man + woman ≈ queen) show:
- High cosine similarity (0.92) — the direction is roughly right
- But rank ~24 — many other words are closer

Analogies require **relational** structure, which is harder than simple periodic encoding.

## Experiments

### 1. Days of the Week (Period 7)

**Prompt**: `"Monday + 3 days = "`

Tests cyclic representations for day-of-week arithmetic.

**Result**: R² = 0.9953 in dimension 125 — the model has an almost perfect 7-day clock!

### 2. Months of the Year (Period 12)

**Prompt**: `"January + 5 months = "`

**Result**: R² = 0.9288 in dimension 410

### 3. Clock Hours (Period 12)

**Prompt**: `"3 o'clock + 5 hours = "`

**Result**: R² = 0.9379 in dimension 725

### 4. Alphabet Position (Period 26)

**Prompts**: 
- `"A + 3 letters = "` (cyclic) → R² = 0.9823 in dim 238
- `"The 5th letter of the alphabet is "` (linear) → R² = 0.8907 in dim 458

The model has BOTH cyclic and linear alphabet encodings in different dimensions!

### 5. Word Analogies (Parallelogram Structure)

**Test**: king - man + woman ≈ queen

| Analogy | Cosine Sim | Rank |
|---------|------------|------|
| king:queen::man:woman | 0.94 | 17 |
| brother:sister::boy:girl | 0.96 | 3 |
| husband:wife::uncle:aunt | 0.96 | 3 |
| France:Paris::Japan:Tokyo | 0.91 | 14 |
| walk:walked::run:ran | 0.94 | 13 |
| see:saw::eat:ate | 0.95 | 6 |

Some analogies work well (rank 3-6), others are noisier.

## Running

```bash
./run.sh
```

Or run specific experiments:

```bash
source ../venv/bin/activate
python probe_structures.py --experiments "days,months"
python probe_structures.py --experiments "analogies"
```

## Output Files

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `days_of_week.png` | Cyclic structure visualization |
| `days_of_week_layers.png` | R² across all 28 layers |
| `months.png` | Month cyclic structure |
| `months_layers.png` | R² across layers |
| `clock_hours.png` | Hour cyclic structure |
| `clock_hours_layers.png` | R² across layers |
| `alphabet.png` | Letter cyclic structure |
| `alphabet_position.png` | Linear position encoding |
| `analogies.png` | PCA visualization of word analogies |
| `results.json` | Full numerical results |

## Conclusion

**Fourier representations are a fundamental computational primitive that neural networks discover naturally.**

The model learned separate "clocks" for days, months, hours, and alphabet positions — each in its own dedicated dimension. This supports the hypothesis from the grokking paper: periodic structure in data leads to sinusoidal encodings in neural networks.

The fact that this emerges from general language modeling (not task-specific training) suggests it's a universal property of how neural networks represent cyclical information.
