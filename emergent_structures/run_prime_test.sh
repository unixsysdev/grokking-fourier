#!/bin/bash
# Test whether Fourier dimensions generalize across different primes

set -e

cd "$(dirname "$0")"

source ../venv/bin/activate

echo "=============================================="
echo "Testing Prime Generalization"
echo "=============================================="
echo ""
echo "Questions:"
echo "  1. Does dimension 35 generalize across primes?"
echo "  2. Do different primes activate different dimensions?"
echo ""

python test_prime_generalization.py \
    --model "Qwen/Qwen3-0.6B" \
    --primes "7,11,13,17,19,23,29,31" \
    --layer 14 \
    --output_dir "results_primes"

echo ""
echo "=============================================="
echo "Test complete!"
echo "=============================================="
