"""
Clean sine wave emergence visualization.

Shows ONE neuron at a time, with clear before/after comparison.
Much cleaner than the 6-neuron overlay.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from model import OneLayerTransformer, create_modular_addition_data


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
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
    """Get neuron activations organized by (a+b) mod p."""
    p = model.p
    a, b, targets = create_modular_addition_data(p, device)
    
    with torch.no_grad():
        _ = model(a, b)
        mlp_acts = model.cache['mlp_activations'][:, 2, :]  # (p*p, d_mlp)
    
    sums = (a + b) % p
    
    # Mean activation for each sum value
    mean_activations = torch.zeros(p, mlp_acts.shape[1], device=device)
    # Also get all activations grouped by sum for scatter plot
    all_by_sum = [[] for _ in range(p)]
    
    for i in range(len(sums)):
        s = sums[i].item()
        all_by_sum[s].append(mlp_acts[i].cpu().numpy())
    
    for s in range(p):
        mask = (sums == s)
        if mask.sum() > 0:
            mean_activations[s] = mlp_acts[mask].mean(dim=0)
    
    return mean_activations, all_by_sum


def fit_sinusoid(y, p):
    """Fit best sinusoid to data, return params and R²."""
    x = np.arange(p)
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    
    if ss_tot < 1e-10:
        return 0, 0, 0, 0, np.zeros(p)
    
    best_r2 = 0
    best_freq = 1
    best_coeffs = None
    
    for k in range(1, p // 2 + 1):
        angle = 2 * np.pi * k * x / p
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        
        X = np.column_stack([cos_k, sin_k, np.ones(p)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        y_pred = X @ coeffs
        ss_res = ((y - y_pred) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        
        if r2 > best_r2:
            best_r2 = r2
            best_freq = k
            best_coeffs = coeffs
    
    # Generate fitted curve
    angle = 2 * np.pi * best_freq * x / p
    y_fit = best_coeffs[0] * np.cos(angle) + best_coeffs[1] * np.sin(angle) + best_coeffs[2]
    
    amplitude = np.sqrt(best_coeffs[0]**2 + best_coeffs[1]**2)
    
    return best_r2, best_freq, amplitude, best_coeffs[2], y_fit


def find_best_neuron(mean_activations):
    """Find the neuron with highest R²."""
    p = mean_activations.shape[0]
    n_neurons = mean_activations.shape[1]
    
    best_neuron = 0
    best_r2 = 0
    
    acts_np = mean_activations.cpu().numpy()
    
    for n in range(n_neurons):
        r2, _, _, _, _ = fit_sinusoid(acts_np[:, n], p)
        if r2 > best_r2:
            best_r2 = r2
            best_neuron = n
    
    return best_neuron, best_r2


def generate_clean_frame(checkpoint_path, output_path, history, frame_idx, device, 
                         track_neuron=None):
    """
    Generate a clean frame showing ONE neuron's sine wave emergence.
    """
    model, config, epoch = load_checkpoint(checkpoint_path, device)
    p = config['p']
    
    mean_acts, all_by_sum = get_neuron_activations_by_sum(model, device)
    mean_acts_np = mean_acts.cpu().numpy()
    
    # Find best neuron or use tracked one
    if track_neuron is None:
        neuron_idx, _ = find_best_neuron(mean_acts)
    else:
        neuron_idx = track_neuron
    
    # Get this neuron's data
    y = mean_acts_np[:, neuron_idx]
    r2, freq, amplitude, offset, y_fit = fit_sinusoid(y, p)
    
    # Get scatter data for this neuron
    scatter_x = []
    scatter_y = []
    for s in range(p):
        for act in all_by_sum[s]:
            scatter_x.append(s)
            scatter_y.append(act[neuron_idx])
    
    # Create figure
    fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    
    # Style
    text_color = '#e6edf3'
    grid_color = '#30363d'
    accent_color = '#58a6ff'
    fit_color = '#f78166'
    scatter_color = '#7ee787'
    
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    # Main plot - sine wave
    ax1 = fig.add_axes([0.08, 0.35, 0.6, 0.55])
    ax1.set_facecolor('#161b22')
    
    x = np.arange(p)
    
    # Scatter: all individual activations (faint)
    ax1.scatter(scatter_x, scatter_y, c=scatter_color, alpha=0.15, s=8, label='All samples')
    
    # Mean activations (dots)
    ax1.scatter(x, y, c=accent_color, s=40, zorder=3, label='Mean activation', edgecolors='white', linewidths=0.5)
    
    # Fitted sine wave (thick line)
    if r2 > 0.1:
        ax1.plot(x, y_fit, color=fit_color, linewidth=3, zorder=2, 
                label=f'Sine fit (k={freq}, R²={r2:.3f})')
    
    ax1.set_xlabel('(a + b) mod p', fontsize=12)
    ax1.set_ylabel('Neuron Activation', fontsize=12)
    ax1.set_title(f'Neuron {neuron_idx}', fontsize=14, fontweight='bold', color=text_color)
    ax1.legend(loc='upper right', facecolor='#161b22', edgecolor=grid_color, fontsize=10)
    ax1.grid(True, alpha=0.3, color=grid_color)
    ax1.set_xlim(-2, p+2)
    
    # R² meter (right side)
    ax_meter = fig.add_axes([0.72, 0.35, 0.08, 0.55])
    ax_meter.set_facecolor('#161b22')
    
    # Draw R² bar
    bar_height = r2
    colors = plt.cm.RdYlGn(r2)  # Red to green based on R²
    ax_meter.barh([0], [bar_height], height=0.6, color=colors, edgecolor='white', linewidth=2)
    ax_meter.set_xlim(0, 1)
    ax_meter.set_ylim(-0.5, 0.5)
    ax_meter.set_xlabel('R²', fontsize=12)
    ax_meter.set_yticks([])
    ax_meter.axvline(x=0.8, color='white', linestyle='--', alpha=0.5)
    ax_meter.text(r2 + 0.02, 0, f'{r2:.3f}', va='center', fontsize=14, fontweight='bold', color=text_color)
    
    # Training progress (bottom left)
    ax2 = fig.add_axes([0.08, 0.08, 0.35, 0.2])
    ax2.set_facecolor('#161b22')
    
    epochs = history['epochs'][:frame_idx + 1]
    test_acc = history['test_acc'][:frame_idx + 1]
    
    ax2.fill_between(epochs, test_acc, alpha=0.3, color=scatter_color)
    ax2.plot(epochs, test_acc, color=scatter_color, linewidth=2)
    ax2.axvline(x=epoch, color='white', linestyle='--', alpha=0.7)
    ax2.scatter([epoch], [test_acc[-1]], color='white', s=100, zorder=5)
    
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Test Acc', fontsize=10)
    ax2.set_xlim(0, history['config']['n_epochs'])
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, color=grid_color)
    
    # Stats panel (bottom right)
    ax3 = fig.add_axes([0.5, 0.08, 0.3, 0.2])
    ax3.set_facecolor('#161b22')
    ax3.axis('off')
    
    stats_text = f"""
    Epoch: {epoch:,}
    
    Frequency k = {freq}
    Amplitude = {amplitude:.2f}
    R² = {r2:.4f}
    
    Test Accuracy: {test_acc[-1]:.1%}
    """
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace', color=text_color)
    
    # Main title
    fig.suptitle(f'Sine Wave Emergence | Epoch {epoch:,}', 
                 fontsize=18, fontweight='bold', color='white', y=0.96)
    
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return neuron_idx, r2, freq


def generate_all_clean_frames(checkpoints_dir, history_path, output_dir, device='cuda'):
    """Generate clean frames tracking one consistent neuron."""
    
    checkpoints_path = Path(checkpoints_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    checkpoint_files = sorted(checkpoints_path.glob("checkpoint_e*.pt"))
    final_checkpoint = checkpoints_path / "checkpoint_final.pt"
    if final_checkpoint.exists() and final_checkpoint not in checkpoint_files:
        checkpoint_files.append(final_checkpoint)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # First pass: find which neuron is best at the END of training
    print("Finding best neuron from final checkpoint...")
    model, config, _ = load_checkpoint(checkpoint_files[-1], device)
    mean_acts, _ = get_neuron_activations_by_sum(model, device)
    track_neuron, final_r2 = find_best_neuron(mean_acts)
    print(f"Tracking Neuron {track_neuron} (final R²={final_r2:.3f})")
    
    history_epochs = history['epochs']
    
    for i, cp_path in enumerate(tqdm(checkpoint_files, desc="Generating frames")):
        cp = torch.load(cp_path, map_location='cpu')
        cp_epoch = cp.get('epoch', 0)
        
        frame_idx = min(range(len(history_epochs)), 
                       key=lambda j: abs(history_epochs[j] - cp_epoch))
        
        output_file = output_path / f"frame_{i:05d}.png"
        generate_clean_frame(cp_path, output_file, history, frame_idx, device, 
                           track_neuron=track_neuron)
    
    print(f"\nFrames saved to {output_path}")
    print(f"Tracked Neuron {track_neuron}")
    print(f"\nTo create video:")
    print(f"  ffmpeg -framerate 4 -i {output_path}/frame_%05d.png -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v mpeg4 -q:v 3 sine_clean.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=str)
    parser.add_argument("history_path", type=str)
    parser.add_argument("--output_dir", type=str, default="frames_clean")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    generate_all_clean_frames(
        args.checkpoints_dir,
        args.history_path,
        args.output_dir,
        args.device
    )
