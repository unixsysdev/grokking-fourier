#!/bin/bash
# Quick run script for the grokking Fourier experiment
#
# Supports: NVIDIA CUDA, AMD ROCm, Apple MPS, CPU

set -e

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

# Show device info
echo "=== Device Information ==="
python device_utils.py
echo ""

echo "=== Step 1: Training the model ==="
echo "Training on modular addition mod 113 (paper's main experiment)"
echo ""

python train.py \
    --p 113 \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 512 \
    --train_frac 0.3 \
    --lr 1e-3 \
    --weight_decay 1.0 \
    --n_epochs 25000 \
    --log_every 100 \
    --save_every 2500 \
    --output_dir checkpoints_p113

echo ""
echo "=== Step 2: Running Fourier analysis ==="
python analyze.py checkpoints_p113/checkpoint_final.pt --output_dir analysis_p113

echo ""
echo "=== Done! ==="
echo "Check the 'analysis' folder for the Fourier plots"
echo "Key files:"
echo "  - analysis/training_curves.png    - Loss and accuracy over training"
echo "  - analysis/embedding_fourier.png  - Fourier structure of embeddings"
echo "  - analysis/neuron_logit_fourier.png - Fourier structure of W_L"
echo "  - analysis/neuron_activations.png - Periodicity in neurons"
echo "  - analysis/attention_patterns.png - Periodicity in attention"
echo "  - analysis/logits_2d_fourier.png  - 2D Fourier of logits"
