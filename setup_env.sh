#!/bin/bash
# Setup script for grokking-fourier on AMD Strix Halo (ROCm)
#
# Prerequisites:
#   - ROCm 6.2+ installed (https://rocm.docs.amd.com/)
#   - Python 3.10+ recommended
#
# The Strix Halo uses RDNA 3.5 architecture (gfx1151)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Grokking Fourier Environment Setup ==="
echo "Target: AMD Strix Halo (ROCm)"
echo ""

# Check for ROCm
if command -v rocminfo &> /dev/null; then
    echo "✓ ROCm detected"
    rocminfo | grep -E "Name:|Marketing Name:" | head -4
else
    echo "⚠ ROCm not found in PATH"
    echo "  Install ROCm or ensure it's in your PATH"
    echo "  See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate
echo "✓ Activated venv ($(python --version))"
echo ""

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with ROCm support
echo ""
echo "=== Installing PyTorch with ROCm 6.2 support ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify PyTorch ROCm installation
echo ""
echo "=== Verifying PyTorch Installation ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')  # ROCm uses cuda API
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    # Quick test
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print(f'✓ GPU compute test passed')
else:
    print('⚠ GPU not available, will use CPU')
"

# Install remaining dependencies
echo ""
echo "=== Installing remaining dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run training:"
echo "  python train.py --p 71 --n_epochs 20000"
echo ""
echo "To run analysis:"
echo "  python analyze.py checkpoints/checkpoint_final.pt"
