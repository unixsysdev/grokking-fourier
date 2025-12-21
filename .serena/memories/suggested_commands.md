# Suggested Commands for grokking-fourier

## Environment Setup
```bash
# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies (no requirements.txt, manual install)
pip install torch numpy matplotlib tqdm einops transformers
```

## Running Experiments

### Root Directory (Small Transformer)
```bash
# Full experiment (train + analyze)
./run.sh

# Train with custom parameters
python train.py --p 113 --n_epochs 25000 --output_dir checkpoints_p113

# Analyze a trained model
python analyze.py checkpoints_p113/checkpoint_final.pt --output_dir analysis_p113
```

### MIRAS Experiments (`miras_experiment/`)
```bash
cd miras_experiment

# Cross-entropy supervised training
python train_ce.py

# Reinforcement learning training
python train_rl.py

# Sweep accuracy across primes
python sweep_accuracy.py

# Mechanistic analysis
python analyze_miras_mechanics.py
```

### Qwen3 Analysis (`qwen3_analysis/`)
```bash
cd qwen3_analysis

# Basic analysis
./run.sh

# Deep analysis
./run_deep.sh

# Or run directly
python analyze_qwen3.py
python analyze_deep.py
```

### Emergent Structures (`emergent_structures/`)
```bash
cd emergent_structures

# Main run
./run.sh

# Investigation
./run_investigation.sh

# Prime generalization test
./run_prime_test.sh

# Causal ablation
python causal_ablation.py
```

## Utility Commands (macOS/Darwin)
```bash
# List files
ls -la

# Find Python files
find . -name "*.py" -type f

# Search in code
grep -r "pattern" --include="*.py"

# Git operations
git status
git diff
git add .
git commit -m "message"
```

## Common Parameters
- `--p`: Prime modulus (113 recommended for main experiment)
- `--d_model`: Model dimension (default: 128)
- `--n_heads`: Number of attention heads (default: 4)
- `--d_mlp`: MLP hidden dimension (default: 512)
- `--train_frac`: Fraction for training (default: 0.3)
- `--weight_decay`: Critical for grokking (1.0 for p≥113, 5.0 for smaller)
- `--n_epochs`: Training epochs (grokking ~10k-15k epochs)
