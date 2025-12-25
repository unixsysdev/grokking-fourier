#!/bin/bash
# Run a single phase transition experiment with animation

set -e

# Default parameters
P=${P:-113}
TRAIN_FRAC=${TRAIN_FRAC:-0.3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1.0}
N_EPOCHS=${N_EPOCHS:-30000}
RUN_NAME=${RUN_NAME:-"default"}

echo "=============================================="
echo "Phase Transition Experiment"
echo "=============================================="
echo "p=$P, train_frac=$TRAIN_FRAC, weight_decay=$WEIGHT_DECAY"
echo "n_epochs=$N_EPOCHS, run_name=$RUN_NAME"
echo ""

# Step 1: Train with metrics
echo "[1/3] Training model..."
python train_with_metrics.py \
    --p $P \
    --train_frac $TRAIN_FRAC \
    --weight_decay $WEIGHT_DECAY \
    --n_epochs $N_EPOCHS \
    --checkpoint_every 1000 \
    --compute_metrics_every 100 \
    --run_name $RUN_NAME

# Step 2: Generate frames
echo ""
echo "[2/3] Generating animation frames..."
python generate_frames.py \
    checkpoints/$RUN_NAME/history.json \
    --output_dir frames/$RUN_NAME \
    --skip_every 1

# Step 3: Create video
echo ""
echo "[3/3] Creating video..."
python make_video.py \
    frames/$RUN_NAME \
    --output grokking_${RUN_NAME}.mp4 \
    --framerate 30

echo ""
echo "=============================================="
echo "Done! Video saved to: grokking_${RUN_NAME}.mp4"
echo "=============================================="
