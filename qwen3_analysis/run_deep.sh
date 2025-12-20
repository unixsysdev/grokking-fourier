#!/bin/bash
# Deep analysis of Fourier-like dimensions in Qwen3

set -e

cd "$(dirname "$0")"

# Activate the parent venv
source ../venv/bin/activate

echo "=============================================="
echo "Qwen3 Deep Fourier Analysis"
echo "=============================================="
echo ""
echo "This script performs detailed analysis of"
echo "dimensions that show Fourier-like structure."
echo ""

# Analyze the interesting dimensions we found:
# - Dim 35: appeared in ALL layers as top Fourier dimension
# - Dim 8, 7, 1: also appeared frequently
# - Dim 243: appeared in early layers

python analyze_deep.py \
    --model "Qwen/Qwen3-0.6B" \
    --p 23 \
    --dims "35,8,7,1,243" \
    --output_dir results_detailed

echo ""
echo "=============================================="
echo "Deep analysis complete!"
echo "=============================================="
echo ""
echo "Results saved to qwen3_analysis/results_detailed/"
echo "Key files:"
echo "  - dim*_activation_patterns.png  - 2D activation heatmaps across layers"
echo "  - dim*_sum_activations.png      - Activation vs (a+b) mod p"
echo "  - dim*_metrics_across_layers.png - R², Gini, frequency across layers"
echo "  - dim*_fft_heatmap.png          - FFT magnitude heatmap"
echo "  - all_dims_comparison.png       - Compare all dimensions"
echo "  - detailed_analysis.json        - Full numerical results"
