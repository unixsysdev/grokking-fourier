#!/bin/bash
# Deep investigation of modular arithmetic circuits

set -e

cd "$(dirname "$0")"

source ../venv/bin/activate

echo "=============================================="
echo "Deep Investigation of Modular Circuits"
echo "=============================================="
echo ""
echo "Experiments:"
echo "  1. Why does dim 867 handle primes 17-29?"
echo "  2. How does attention route different moduli?"
echo "  3. Do composite numbers (10,12,24,60,100) reuse cyclic dims?"
echo ""

python investigate_circuits.py \
    --model "Qwen/Qwen3-0.6B" \
    --output_dir "results_deep" \
    --experiments "all"

echo ""
echo "=============================================="
echo "Investigation complete!"
echo "=============================================="
