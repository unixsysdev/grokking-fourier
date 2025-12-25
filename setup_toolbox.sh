#!/bin/bash
# Setup script for grokking-fourier on AMD Strix Halo
#
# Uses the pre-built toolbox image with ROCm 7 + PyTorch already configured
# Image: docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest

set -e

TOOLBOX_NAME="grokking-fourier"
IMAGE="docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest"

echo "=== Grokking Fourier - Strix Halo Setup ==="
echo ""

# Check if toolbox exists
if toolbox list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
    echo "Toolbox '$TOOLBOX_NAME' already exists."
    echo "To enter: toolbox enter $TOOLBOX_NAME"
    echo "To remove and recreate: toolbox rm $TOOLBOX_NAME"
    exit 0
fi

echo "Creating toolbox with ROCm 7 environment..."
echo "Image: $IMAGE"
echo ""

# Create the toolbox with GPU access
toolbox create "$TOOLBOX_NAME" \
    --image "$IMAGE" \
    -- --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined

echo ""
echo "=== Toolbox created successfully! ==="
echo ""
echo "To enter the environment:"
echo "  toolbox enter $TOOLBOX_NAME"
echo ""
echo "Once inside, install project dependencies:"
echo "  cd $(pwd)"
echo "  pip install einops  # Only missing dep from the image"
echo ""
echo "Then run experiments:"
echo "  python device_utils.py  # Verify GPU"
echo "  python train.py --p 71 --n_epochs 20000"
echo ""
echo "Note: Your home directory is shared with the toolbox,"
echo "so this project folder is accessible from inside."
