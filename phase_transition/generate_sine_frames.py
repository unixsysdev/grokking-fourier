"""
Generate frames showing sine wave emergence in neuron activations.

This visualizes the KEY insight: neurons learn to compute sin/cos of (a+b) mod p.
We watch this periodic structure emerge over training.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from model import OneLayerTransformer, create_modular_addition_data


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config - try different locations
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {'p': 113, 'd_model': 128, 'n_heads': 4, 'd_mlp': 512}
    
    model = OneLayerTransformer(
        p=config['p'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_mlp=config['d_mlp'],
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, checkpoint.get('epoch', 0)


def get_neuron_activations_by_sum(model, device):
    """
    Get neuron activations organized by (a+b) mod p.
    
    Returns:
        activations: (p, n_samples_per_sum, d_mlp) - activations grouped by sum
        mean_activations: (p, d_mlp) - mean activation for each sum value
    """
    p = model.p
    
    # Create all input pairs
    a, b, targets = create_modular_addition_data(p, device)
    
    # Forward pass
    with torch.no_grad():
        _ = model(a, b)
        mlp_acts = model.cache['mlp_activations'][:, 2, :]  # (p*p, d_mlp)
    
    # Group by (a+b) mod p
    sums = (a + b) % p
    
    # Compute mean activation for each sum value
    mean_activations = torch.zeros(p, mlp_acts.shape[1], device=device)
    for s in range(p):
        mask = (sums == s)
        if mask.sum() > 0:
            mean_activations[s] = mlp_acts[mask].mean(dim=0)
    
    return mlp_acts, mean_activations, sums


def compute_sinusoid_fit(activations, p):
    """
    Fit sinusoids to activation pattern and return R² for each neuron.
    
    For each neuron, finds the best-fitting frequency k and computes R².
    """
    x = np.arange(p)
    n_neurons = activations.shape[1]
    
    best_r2 = np.zeros(n_neurons)
    best_freq = np.zeros(n_neurons, dtype=int)
    best_phase = np.zeros(n_neurons)
    
    acts_np = activations.cpu().numpy()
    
    for neuron_idx in range(n_neurons):
        y = acts_np[:, neuron_idx]
        y_mean = y.mean()
        ss_tot = ((y - y_mean) ** 2).sum()
        
        if ss_tot < 1e-10:
            continue
        
        # Try each frequency
        for k in range(1, p // 2 + 1):
            angle = 2 * np.pi * k * x / p
            
            # Fit A*cos(angle) + B*sin(angle) + C
            cos_k = np.cos(angle)
            sin_k = np.sin(angle)
            
            # Least squares fit
            X = np.column_stack([cos_k, sin_k, np.ones(p)])
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            y_pred = X @ coeffs
            ss_res = ((y - y_pred) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            
            if r2 > best_r2[neuron_idx]:
                best_r2[neuron_idx] = r2
                best_freq[neuron_idx] = k
                best_phase[neuron_idx] = np.arctan2(coeffs[1], coeffs[0])
    
    return best_r2, best_freq, best_phase


def generate_sine_frame(checkpoint_path, output_path, history, frame_idx, device):
    """
    Generate a frame showing sine wave emergence.
    
    Layout:
    ┌─────────────────────────────────────────────────┐
    │  Top Neurons: Activation vs (a+b) mod p         │
    │  (showing individual neurons + sine fits)       │
    ├─────────────────────────────────────────────────┤
    │  Left: R² distribution  │  Right: Loss/Acc     │
    ├─────────────────────────────────────────────────┤
    │  Frequency spectrum (which k values are used)   │
    └─────────────────────────────────────────────────┘
    """
    # Load model
    model, config, epoch = load_checkpoint(checkpoint_path, device)
    p = config['p']
    
    # Get activations
    all_acts, mean_acts, sums = get_neuron_activations_by_sum(model, device)
    
    # Compute sinusoid fits
    r2_scores, best_freqs, best_phases = compute_sinusoid_fit(mean_acts, p)
    
    # Find top neurons by R²
    top_k = 6
    top_neurons = np.argsort(r2_scores)[-top_k:][::-1]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 0.8], hspace=0.3, wspace=0.25)
    
    # Style
    text_color = '#e0e0e0'
    grid_color = '#3a3a5a'
    
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 1: Top neurons with sine fits (top, full width)
    # ═══════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#16213e')
    
    x = np.arange(p)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, top_k))
    
    mean_acts_np = mean_acts.cpu().numpy()
    
    for i, neuron_idx in enumerate(top_neurons):
        y = mean_acts_np[:, neuron_idx]
        r2 = r2_scores[neuron_idx]
        freq = best_freqs[neuron_idx]
        
        # Plot actual activations
        ax1.plot(x, y, 'o', color=colors[i], alpha=0.6, markersize=3,
                label=f'Neuron {neuron_idx} (k={freq}, R²={r2:.3f})')
        
        # Plot best-fit sinusoid
        if r2 > 0.1:
            angle = 2 * np.pi * freq * x / p
            cos_k = np.cos(angle)
            sin_k = np.sin(angle)
            X = np.column_stack([cos_k, sin_k, np.ones(p)])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_fit = X @ coeffs
            ax1.plot(x, y_fit, '-', color=colors[i], linewidth=2, alpha=0.9)
    
    ax1.set_xlabel('(a + b) mod p', fontsize=12)
    ax1.set_ylabel('Mean Activation', fontsize=12)
    ax1.set_title(f'Top {top_k} Periodic Neurons — Epoch {epoch:,}', 
                  fontsize=14, fontweight='bold', color=text_color)
    ax1.legend(loc='upper right', facecolor='#16213e', edgecolor=grid_color, 
               fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, color=grid_color)
    ax1.set_xlim(0, p-1)
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 2a: R² distribution (middle-left)
    # ═══════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#16213e')
    
    # Histogram of R² scores
    bins = np.linspace(0, 1, 30)
    ax2.hist(r2_scores, bins=bins, color='#00d4ff', alpha=0.7, edgecolor='white')
    
    # Mark threshold for "periodic" neurons
    ax2.axvline(x=0.8, color='#00ff88', linestyle='--', linewidth=2, label='R² > 0.8 threshold')
    
    n_periodic = (r2_scores > 0.8).sum()
    ax2.text(0.82, ax2.get_ylim()[1] * 0.9, f'{n_periodic} neurons', 
             color='#00ff88', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('R² (sinusoid fit quality)', fontsize=11)
    ax2.set_ylabel('Number of neurons', fontsize=11)
    ax2.set_title('Neuron Periodicity Distribution', fontsize=13, fontweight='bold', color=text_color)
    ax2.legend(loc='upper left', facecolor='#16213e', edgecolor=grid_color)
    ax2.grid(True, alpha=0.3, color=grid_color)
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 2b: Loss and accuracy curves (middle-right)
    # ═══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#16213e')
    
    epochs = history['epochs'][:frame_idx + 1]
    train_loss = history['train_loss'][:frame_idx + 1]
    test_loss = history['test_loss'][:frame_idx + 1]
    test_acc = history['test_acc'][:frame_idx + 1]
    
    ax3.semilogy(epochs, train_loss, color='#00d4ff', linewidth=2, label='Train Loss', alpha=0.8)
    ax3.semilogy(epochs, test_loss, color='#ff6b6b', linewidth=2, label='Test Loss', alpha=0.8)
    
    # Secondary axis for accuracy
    ax3b = ax3.twinx()
    ax3b.plot(epochs, test_acc, color='#00ff88', linewidth=2.5, label='Test Acc')
    ax3b.set_ylabel('Test Accuracy', fontsize=11, color='#00ff88')
    ax3b.tick_params(axis='y', colors='#00ff88')
    ax3b.set_ylim(0, 1.05)
    
    # Mark current epoch
    ax3.axvline(x=epoch, color='white', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss (log)', fontsize=11)
    ax3.set_title('Training Progress', fontsize=13, fontweight='bold', color=text_color)
    ax3.legend(loc='upper right', facecolor='#16213e', edgecolor=grid_color)
    ax3.grid(True, alpha=0.3, color=grid_color)
    ax3.set_xlim(0, history['config']['n_epochs'])
    
    # ═══════════════════════════════════════════════════════════════
    # Panel 3: Frequency usage histogram (bottom, full width)
    # ═══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#16213e')
    
    # Count neurons per frequency (only for neurons with R² > 0.5)
    periodic_mask = r2_scores > 0.5
    freq_counts = np.zeros(p // 2 + 1)
    for freq in best_freqs[periodic_mask]:
        freq_counts[freq] += 1
    
    # Weight by R² score
    freq_weights = np.zeros(p // 2 + 1)
    for i, (freq, r2) in enumerate(zip(best_freqs, r2_scores)):
        if r2 > 0.5:
            freq_weights[freq] += r2
    
    freqs = np.arange(1, p // 2 + 1)
    bars = ax4.bar(freqs, freq_weights[1:], color='#ff9f43', alpha=0.8, width=0.8)
    
    # Highlight top frequencies
    top_freqs = np.argsort(freq_weights[1:])[-5:][::-1] + 1
    for tf in top_freqs:
        if freq_weights[tf] > 0:
            ax4.bar([tf], [freq_weights[tf]], color='#00ff88', alpha=1.0, width=0.8)
    
    ax4.set_xlabel('Frequency k', fontsize=11)
    ax4.set_ylabel('Weighted count (Σ R²)', fontsize=11)
    ax4.set_title(f'Key Frequencies: {list(top_freqs[:3])}', fontsize=13, fontweight='bold', color=text_color)
    ax4.grid(True, alpha=0.3, color=grid_color, axis='y')
    ax4.set_xlim(0, min(60, p // 2))
    
    # ═══════════════════════════════════════════════════════════════
    # Title and info
    # ═══════════════════════════════════════════════════════════════
    fig.suptitle(
        f'🌊 Sine Wave Emergence in Grokking | Epoch {epoch:,}',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )
    
    # Stats
    mean_r2 = r2_scores.mean()
    max_r2 = r2_scores.max()
    info_text = f'Mean R²={mean_r2:.3f} | Max R²={max_r2:.3f} | Periodic neurons (R²>0.8): {n_periodic}'
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, color='#888888')
    
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return {
        'epoch': epoch,
        'mean_r2': mean_r2,
        'max_r2': max_r2,
        'n_periodic': n_periodic,
        'top_freqs': top_freqs.tolist(),
    }


def generate_all_sine_frames(checkpoints_dir, history_path, output_dir, device='cuda'):
    """Generate all frames from checkpoints."""
    
    checkpoints_path = Path(checkpoints_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Find all checkpoints
    checkpoint_files = sorted(checkpoints_path.glob("checkpoint_e*.pt"))
    
    # Add final checkpoint
    final_checkpoint = checkpoints_path / "checkpoint_final.pt"
    if final_checkpoint.exists() and final_checkpoint not in checkpoint_files:
        checkpoint_files.append(final_checkpoint)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    print(f"Output directory: {output_path}")
    
    # Map epochs to history indices
    history_epochs = history['epochs']
    
    stats = []
    for i, cp_path in enumerate(tqdm(checkpoint_files, desc="Generating sine frames")):
        # Find corresponding history index
        cp = torch.load(cp_path, map_location='cpu')
        cp_epoch = cp.get('epoch', 0)
        
        # Find closest history index
        frame_idx = min(range(len(history_epochs)), 
                       key=lambda j: abs(history_epochs[j] - cp_epoch))
        
        output_file = output_path / f"frame_{i:05d}.png"
        frame_stats = generate_sine_frame(cp_path, output_file, history, frame_idx, device)
        stats.append(frame_stats)
    
    # Save stats (convert numpy types to native Python)
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    with open(output_path / "frame_stats.json", "w") as f:
        json.dump(convert_to_native(stats), f, indent=2)
    
    print(f"\nFrames saved to {output_path}")
    print(f"To create video:")
    print(f"  ffmpeg -framerate 5 -i {output_path}/frame_%05d.png -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p sine_emergence.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sine wave emergence animation")
    parser.add_argument("checkpoints_dir", type=str, help="Directory with checkpoints")
    parser.add_argument("history_path", type=str, help="Path to history.json")
    parser.add_argument("--output_dir", type=str, default="frames_sine",
                        help="Output directory for frames")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    generate_all_sine_frames(
        args.checkpoints_dir,
        args.history_path,
        args.output_dir,
        args.device
    )
