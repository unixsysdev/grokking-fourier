#!/bin/bash
# Run Qwen3 Fourier analysis experiments

set -e

cd "$(dirname "$0")"

# Activate the parent venv
source ../venv/bin/activate

echo "=============================================="
echo "Qwen3 0.6B Fourier Analysis for Modular Arithmetic"
echo "=============================================="
echo ""
echo "This experiment tests whether a pretrained LLM"
echo "shows Fourier structure when doing modular arithmetic,"
echo "similar to what was found in the grokking paper."
echo ""

# Use a small prime for faster analysis (23^2 = 529 pairs)
# Qwen3 0.6B has 28 layers, analyze a spread
python analyze_qwen3.py \
    --model "Qwen/Qwen3-0.6B" \
    --p 23 \
    --layers "3,7,14,21,27" \
    --activation_type hidden \
    --output_dir results

echo ""
echo "=============================================="
echo "Analysis complete!"
echo "=============================================="
echo ""
echo "Results saved to qwen3_analysis/results/"
echo "Key files:"
echo "  - results/summary.json - Numerical results"
echo "  - results/fourier_analysis_layer*_hidden.png - 2D FFT plots"
echo "  - results/top_dims_layer*_hidden.png - Top Fourier-like dimensions"
