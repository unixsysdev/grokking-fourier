"""
Generate animation frames from training history.

Creates beautiful visualizations showing the evolution of:
- Loss curves
- Fourier signal strength
- Embedding geometry
- Frequency spectrum
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import argparse


def load_history(history_path):
    """Load training history from JSON."""
    with open(history_path, 'r') as f:
        return json.load(f)


def generate_frame(history, frame_idx, output_path, total_frames):
    """
    Generate a single frame of the animation.
    
    Layout:
    ┌─────────────────┬─────────────────┐
    │  Loss Curves    │  Accuracy       │
    ├─────────────────┼─────────────────┤
    │  Fourier        │  Embedding      │
    │  Strength       │  Geometry       │
    ├─────────────────┴─────────────────┤
    │     Frequency Spectrum            │
    └───────────────────────────────────┘
    """
    # Extract data up to this frame
    epochs = history['epochs'][:frame_idx + 1]
    train_loss = history['train_loss'][:frame_idx + 1]
    test_loss = history['test_loss'][:frame_idx + 1]
    train_acc = history['train_acc'][:frame_idx + 1]
    test_acc = history['test_acc'][:frame_idx + 1]
    fourier_strength = history['fourier_strength'][:frame_idx + 1]
    weight_norm = history['weight_norm'][:frame_idx + 1]
    pca_2d = history['pca_2d'][frame_idx]  # Current embedding
    freq_spectrum = history['freq_spectrum'][frame_idx]  # Current spectrum
    
    current_epoch = epochs[-1]
    config = history['config']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8], 
                  hspace=0.3, wspace=0.25)
    
    # Style settings
    text_color = '#e0e0e0'
    grid_color = '#3a3a5a'
    train_color = '#00d4ff'
    test_color = '#ff6b6b'
    fourier_color = '#00ff88'
    
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 1: Loss Curves (top-left)
    # ═══════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    
    ax1.semilogy(epochs, train_loss, color=train_color, linewidth=2, label='Train', alpha=0.9)
    ax1.semilogy(epochs, test_loss, color=test_color, linewidth=2, label='Test', alpha=0.9)
    
    # Mark current position
    ax1.scatter([current_epoch], [train_loss[-1]], color=train_color, s=100, zorder=5)
    ax1.scatter([current_epoch], [test_loss[-1]], color=test_color, s=100, zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (log)', fontsize=11)
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold', color=text_color)
    ax1.legend(loc='upper right', facecolor='#16213e', edgecolor=grid_color)
    ax1.grid(True, alpha=0.3, color=grid_color)
    ax1.set_xlim(0, history['config']['n_epochs'])
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 2: Accuracy (top-right)
    # ═══════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    ax2.plot(epochs, train_acc, color=train_color, linewidth=2, label='Train', alpha=0.9)
    ax2.plot(epochs, test_acc, color=test_color, linewidth=2, label='Test', alpha=0.9)
    
    # Mark current position
    ax2.scatter([current_epoch], [train_acc[-1]], color=train_color, s=100, zorder=5)
    ax2.scatter([current_epoch], [test_acc[-1]], color=test_color, s=100, zorder=5)
    
    # Grokking threshold line
    ax2.axhline(y=0.95, color='#ffcc00', linestyle='--', alpha=0.5, label='95% threshold')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy', fontsize=13, fontweight='bold', color=text_color)
    ax2.legend(loc='lower right', facecolor='#16213e', edgecolor=grid_color)
    ax2.grid(True, alpha=0.3, color=grid_color)
    ax2.set_xlim(0, history['config']['n_epochs'])
    ax2.set_ylim(0, 1.05)
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 3: Fourier Strength + Weight Norm (middle-left)
    # ═══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#16213e')
    
    ax3.plot(epochs, fourier_strength, color=fourier_color, linewidth=2.5, label='Fourier Strength')
    ax3.scatter([current_epoch], [fourier_strength[-1]], color=fourier_color, s=120, zorder=5)
    
    # Secondary axis for weight norm
    ax3b = ax3.twinx()
    ax3b.plot(epochs, weight_norm, color='#ff9f43', linewidth=2, alpha=0.7, label='Weight Norm')
    ax3b.tick_params(axis='y', colors='#ff9f43')
    ax3b.set_ylabel('Weight Norm', fontsize=11, color='#ff9f43')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Fourier Strength', fontsize=11, color=fourier_color)
    ax3.set_title('Fourier Circuit Formation', fontsize=13, fontweight='bold', color=text_color)
    ax3.tick_params(axis='y', colors=fourier_color)
    ax3.grid(True, alpha=0.3, color=grid_color)
    ax3.set_xlim(0, history['config']['n_epochs'])
    ax3.set_ylim(0, 1.0)
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 4: Embedding Geometry (middle-right)
    # ═══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    
    pca_arr = np.array(pca_2d)
    
    # Color by position (0 to p-1)
    p = config['p']
    colors = plt.cm.hsv(np.linspace(0, 1, p))
    
    ax4.scatter(pca_arr[:, 0], pca_arr[:, 1], c=colors, s=30, alpha=0.8)
    
    # Draw lines connecting consecutive points to show the ordering
    for i in range(p):
        next_i = (i + 1) % p
        ax4.plot([pca_arr[i, 0], pca_arr[next_i, 0]], 
                 [pca_arr[i, 1], pca_arr[next_i, 1]], 
                 color=colors[i], alpha=0.3, linewidth=0.5)
    
    # Add reference circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.max(np.abs(pca_arr)) * 1.1
    ax4.plot(r * np.cos(theta), r * np.sin(theta), '--', color=grid_color, alpha=0.5)
    
    ax4.set_xlabel('PCA 1', fontsize=11)
    ax4.set_ylabel('PCA 2', fontsize=11)
    ax4.set_title(f'Embedding Geometry (p={p})', fontsize=13, fontweight='bold', color=text_color)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.2, color=grid_color)
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 5: Frequency Spectrum (bottom, full width)
    # ═══════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor('#16213e')
    
    n_freqs = len(freq_spectrum)
    freqs = np.arange(1, n_freqs + 1)
    
    # Create gradient colors based on strength
    colors = plt.cm.plasma(np.array(freq_spectrum) / (max(freq_spectrum) + 1e-10))
    
    bars = ax5.bar(freqs, freq_spectrum, color=colors, alpha=0.9, width=0.8)
    
    # Highlight key frequencies
    key_freqs = history['key_frequencies'][frame_idx]
    for kf in key_freqs[:3]:  # Top 3
        if kf <= n_freqs:
            ax5.bar([kf], [freq_spectrum[kf-1]], color=fourier_color, alpha=1.0, width=0.8, 
                   edgecolor='white', linewidth=2)
    
    ax5.set_xlabel('Frequency k', fontsize=11)
    ax5.set_ylabel('Power (normalized)', fontsize=11)
    ax5.set_title(f'Fourier Spectrum (Key frequencies: {key_freqs[:3]})', 
                  fontsize=13, fontweight='bold', color=text_color)
    ax5.grid(True, alpha=0.3, color=grid_color, axis='y')
    ax5.set_xlim(0, min(60, n_freqs + 1))  # Show first 60 frequencies
    
    # ═══════════════════════════════════════════════════════════════
    # Title and info
    # ═══════════════════════════════════════════════════════════════
    fig.suptitle(
        f'Grokking Phase Transition | Epoch {current_epoch:,} / {config["n_epochs"]:,}',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )
    
    # Add config info
    info_text = f'p={config["p"]} | train_frac={config["train_frac"]} | weight_decay={config["weight_decay"]}'
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, color='#888888')
    
    # Progress bar
    progress = (frame_idx + 1) / total_frames
    ax_prog = fig.add_axes([0.1, 0.95, 0.8, 0.015])
    ax_prog.barh([0], [progress], color=fourier_color, height=1)
    ax_prog.barh([0], [1-progress], left=[progress], color='#3a3a5a', height=1)
    ax_prog.set_xlim(0, 1)
    ax_prog.axis('off')
    
    # Save
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def generate_all_frames(history_path, output_dir, skip_every=1):
    """
    Generate all frames for the animation.
    
    Args:
        history_path: Path to history.json
        output_dir: Directory to save frames
        skip_every: Only generate every N-th frame (for faster previews)
    """
    history = load_history(history_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_frames = len(history['epochs'])
    frame_indices = list(range(0, n_frames, skip_every))
    
    print(f"Generating {len(frame_indices)} frames...")
    print(f"Output directory: {output_path}")
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Generating frames")):
        output_file = output_path / f"frame_{i:05d}.png"
        generate_frame(history, frame_idx, output_file, len(frame_indices))
    
    print(f"\nFrames saved to {output_path}")
    print(f"To create video, run:")
    print(f"  ffmpeg -framerate 30 -i {output_path}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p grokking.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animation frames from training history")
    parser.add_argument("history_path", type=str, help="Path to history.json")
    parser.add_argument("--output_dir", type=str, default="phase_transition/frames", 
                        help="Directory to save frames")
    parser.add_argument("--skip_every", type=int, default=1,
                        help="Only generate every N-th frame")
    
    args = parser.parse_args()
    
    generate_all_frames(args.history_path, args.output_dir, args.skip_every)
