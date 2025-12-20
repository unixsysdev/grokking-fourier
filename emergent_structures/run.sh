#!/bin/bash
# Probe for emergent structures in Qwen3

set -e

cd "$(dirname "$0")"

# Activate the venv from parent directory
source ../venv/bin/activate

echo "=============================================="
echo "Probing for Emergent Structures in Qwen3"
echo "=============================================="
echo ""
echo "This script tests whether Qwen3 has developed"
echo "Fourier-like or other structured representations"
echo "for various domains beyond modular arithmetic."
echo ""
echo "Experiments:"
echo "  1. Days of the week (period 7)"
echo "  2. Months of the year (period 12)"
echo "  3. Clock hours (period 12)"
echo "  4. Alphabet positions (period 26)"
echo "  5. Word analogies (parallelogram structure)"
echo ""

python probe_structures.py \
    --model "Qwen/Qwen3-0.6B" \
    --output_dir "results" \
    --experiments "all"

echo ""
echo "=============================================="
echo "Probing complete!"
echo "=============================================="
echo ""
echo "Results saved to emergent_structures/results/"
echo ""
echo "Key files:"
echo "  - days_of_week.png        - Cyclic structure for days"
echo "  - months.png              - Cyclic structure for months"
echo "  - clock_hours.png         - Cyclic structure for hours"
echo "  - alphabet.png            - Cyclic structure for letters"
echo "  - alphabet_position.png   - Linear position encoding"
echo "  - analogies.png           - Parallelogram structure (PCA)"
echo "  - results.json            - Full numerical results"
