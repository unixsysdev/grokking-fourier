"""
Visualize the embedding circle emerging over training.

The key insight: after grokking, the token embeddings for numbers 0, 1, 2, ..., p-1
arrange themselves on a CIRCLE in embedding space. This is because the model
learns to represent numbers as (cos(2πkn/p), sin(2πkn/p)) for key frequencies k.

We project embeddings onto the Fourier basis (cos/sin of dominant frequency) to see
the circle emerge. Standard PCA often misses this structure because the circle may
not lie in the top 2 principal components.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from model import OneLayerTransformer


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


def get_fourier_basis(p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create Fourier basis vectors for a prime p.
    
    Returns:
        cos_basis: (p, p//2+1) - cosine components for each frequency
        sin_basis: (p, p//2+1) - sine components for each frequency
    """
    n = np.arange(p)
    k = np.arange(p // 2 + 1)
    angles = 2 * np.pi * np.outer(n, k) / p  # (p, p//2+1)
    
    cos_basis = np.cos(angles)
    sin_basis = np.sin(angles)
    
    # Normalize
    cos_basis = cos_basis / np.sqrt(p / 2)
    sin_basis = sin_basis / np.sqrt(p / 2)
    
    return cos_basis, sin_basis


def get_embedding_fourier(model, fixed_freq: int = None) -> dict:
    """
    Extract token embeddings and project onto Fourier basis.
    
    Args:
        model: The trained model
        fixed_freq: If provided, use this frequency instead of auto-detecting dominant one.
                   This ensures consistent projection across frames.
    
    Returns dict with:
        fourier_2d: (p, 2) array - projection onto dominant frequency's cos/sin
        circularity: how circular the embedding is in Fourier space
        dominant_freq: the frequency with strongest signal (or fixed_freq if provided)
        freq_strengths: strength of each frequency
        pca_2d: (p, 2) array - standard PCA for comparison
        pca_explained_var: variance explained by top 2 PCA components
    """
    p = model.p
    
    # Get embeddings for number tokens (exclude = token)
    W_E = model.token_embed.weight[:p, :].detach().cpu().numpy()  # (p, d_model)
    
    # Center
    W_E_centered = W_E - W_E.mean(axis=0, keepdims=True)
    
    # Get Fourier basis
    cos_basis, sin_basis = get_fourier_basis(p)
    
    # Project embeddings onto Fourier basis
    # For each frequency k, compute how much the embedding aligns with cos/sin
    cos_coeffs = cos_basis.T @ W_E_centered  # (p//2+1, d_model)
    sin_coeffs = sin_basis.T @ W_E_centered  # (p//2+1, d_model)
    
    # Compute strength of each frequency (sum of squared coefficients over d_model)
    freq_strengths = (cos_coeffs ** 2).sum(axis=1) + (sin_coeffs ** 2).sum(axis=1)
    
    # Skip DC component (k=0), find dominant frequency
    freq_strengths_for_max = freq_strengths.copy()
    freq_strengths_for_max[0] = 0  # Ignore DC
    auto_dominant_freq = np.argmax(freq_strengths_for_max)
    
    # Use fixed frequency if provided, otherwise use auto-detected
    dominant_freq = fixed_freq if fixed_freq is not None else auto_dominant_freq
    
    # Project onto the chosen frequency's cos/sin subspace
    # This gives us a 2D view where the circle should be most visible
    cos_k = cos_basis[:, dominant_freq]  # (p,)
    sin_k = sin_basis[:, dominant_freq]  # (p,)
    
    # Project each embedding dimension onto cos_k and sin_k, then sum
    cos_norm = np.linalg.norm(cos_coeffs[dominant_freq, :])
    sin_norm = np.linalg.norm(sin_coeffs[dominant_freq, :])
    
    if cos_norm > 1e-10:
        x_fourier = W_E_centered @ cos_coeffs[dominant_freq, :].T / cos_norm
    else:
        x_fourier = np.zeros(p)
    
    if sin_norm > 1e-10:
        y_fourier = W_E_centered @ sin_coeffs[dominant_freq, :].T / sin_norm
    else:
        y_fourier = np.zeros(p)
    
    fourier_2d = np.stack([x_fourier, y_fourier], axis=1)  # (p, 2)
    
    # Compute circularity in Fourier space
    center = fourier_2d.mean(axis=0)
    centered = fourier_2d - center
    distances = np.linalg.norm(centered, axis=1)
    
    if distances.mean() > 1e-10:
        cv = distances.std() / distances.mean()
        circularity = 1 - min(cv, 1)
    else:
        circularity = 0
    
    # Also compute PCA for comparison
    U, S, Vh = np.linalg.svd(W_E_centered, full_matrices=False)
    pca_2d = W_E_centered @ Vh[:2, :].T
    total_var = (S ** 2).sum()
    pca_explained_var = (S[:2] ** 2).sum() / total_var if total_var > 0 else 0
    
    # Variance explained by chosen frequency
    total_embedding_var = (W_E_centered ** 2).sum()
    freq_var = freq_strengths[dominant_freq]
    freq_explained_var = freq_var / total_embedding_var if total_embedding_var > 0 else 0
    
    return {
        'fourier_2d': fourier_2d,
        'circularity': circularity,
        'dominant_freq': dominant_freq,
        'auto_dominant_freq': auto_dominant_freq,
        'freq_strengths': freq_strengths,
        'freq_explained_var': freq_explained_var,
        'pca_2d': pca_2d,
        'pca_explained_var': pca_explained_var,
        'singular_values': S,
    }


def get_embedding_pca(model):
    """
    Extract token embeddings and project to 2D via PCA.
    (Kept for backward compatibility)
    
    Returns:
        pca_2d: (p, 2) array of 2D coordinates
        circularity: measure of how circular the embedding is
        explained_var: variance explained by first 2 PCs
    """
    result = get_embedding_fourier(model)
    return (
        result['pca_2d'], 
        result['circularity'],  # Now uses Fourier circularity
        result['pca_explained_var'], 
        result['singular_values']
    )


def compute_fourier_fit(pca_2d, p):
    """
    Measure how well the embedding matches a Fourier circle.
    
    A perfect Fourier embedding would have points at angles 2πkn/p for some k.
    """
    # Center the points
    center = pca_2d.mean(axis=0)
    centered = pca_2d - center
    
    # Convert to angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    
    # For a perfect embedding with frequency k, angle[n] = 2πkn/p + phase
    # Try different k values and find best fit
    best_r2 = 0
    best_k = 1
    
    n = np.arange(p)
    
    for k in range(1, p // 2 + 1):
        # Expected angles (mod 2π)
        expected = (2 * np.pi * k * n / p) % (2 * np.pi)
        
        # Unwrap actual angles to match
        actual_unwrapped = np.unwrap(angles)
        
        # Find best phase alignment
        for phase_offset in np.linspace(0, 2 * np.pi, 20):
            expected_shifted = (expected + phase_offset) % (2 * np.pi)
            
            # Compute circular correlation
            diff = np.minimum(
                np.abs(angles - expected_shifted + np.pi) % (2 * np.pi) - np.pi,
                np.abs(angles - expected_shifted - np.pi) % (2 * np.pi) - np.pi
            )
            
            # Simple R² approximation
            ss_res = (diff ** 2).sum()
            ss_tot = ((angles - angles.mean()) ** 2).sum()
            if ss_tot > 0:
                r2 = max(0, 1 - ss_res / ss_tot)
                if r2 > best_r2:
                    best_r2 = r2
                    best_k = k
    
    return best_k, best_r2


def generate_circle_frame(checkpoint_path, output_path, history, frame_idx, device, fixed_freq=None):
    """Generate a frame showing the embedding circle using Fourier projection.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        output_path: Where to save the frame
        history: Training history dict
        frame_idx: Index into history arrays
        device: torch device
        fixed_freq: If provided, use this Fourier frequency for projection (for consistency across frames)
    """
    
    model, config, epoch = load_checkpoint(checkpoint_path, device)
    p = config['p']
    
    # Get embedding analysis with Fourier projection
    embed_data = get_embedding_fourier(model, fixed_freq=fixed_freq)
    fourier_2d = embed_data['fourier_2d']
    circularity = embed_data['circularity']
    dominant_freq = embed_data['dominant_freq']
    freq_strengths = embed_data['freq_strengths']
    freq_explained_var = embed_data['freq_explained_var']
    
    # Get history data
    epochs = history['epochs'][:frame_idx + 1]
    test_acc = history['test_acc'][:frame_idx + 1]
    
    # Create figure
    fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    
    # Colors
    text_color = '#e6edf3'
    grid_color = '#30363d'
    
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    # === Main plot: Embedding circle (Fourier projection) ===
    ax1 = fig.add_axes([0.05, 0.15, 0.55, 0.75])
    ax1.set_facecolor('#161b22')
    
    # Color points by their value (0 to p-1) using HSV colormap
    colors = plt.cm.hsv(np.linspace(0, 1, p))
    
    # Plot points
    scatter = ax1.scatter(fourier_2d[:, 0], fourier_2d[:, 1], c=colors, s=50, alpha=0.9,
                         edgecolors='white', linewidths=0.3)
    
    # Connect consecutive points with lines to show ordering
    for i in range(p):
        next_i = (i + 1) % p
        ax1.plot([fourier_2d[i, 0], fourier_2d[next_i, 0]], 
                [fourier_2d[i, 1], fourier_2d[next_i, 1]], 
                color=colors[i], alpha=0.4, linewidth=1)
    
    # Mark 0 specially
    ax1.scatter([fourier_2d[0, 0]], [fourier_2d[0, 1]], c='white', s=150, marker='*', 
               zorder=10, edgecolors='black', linewidths=1)
    ax1.annotate('0', (fourier_2d[0, 0], fourier_2d[0, 1]), fontsize=12, fontweight='bold',
                color='white', xytext=(5, 5), textcoords='offset points')
    
    # Draw reference circle
    center = fourier_2d.mean(axis=0)
    distances = np.linalg.norm(fourier_2d - center, axis=1)
    radius = distances.mean()
    circle = Circle(center, radius, fill=False, color='white', linestyle='--', 
                   alpha=0.3, linewidth=2)
    ax1.add_patch(circle)
    
    # Equal aspect ratio
    ax1.set_aspect('equal')
    
    # Limits with padding
    margin = max(radius * 0.3, 0.1)
    ax1.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax1.set_ylim(center[1] - radius - margin, center[1] + radius + margin)
    
    ax1.set_xlabel(f'Fourier cos(k={dominant_freq})', fontsize=11)
    ax1.set_ylabel(f'Fourier sin(k={dominant_freq})', fontsize=11)
    ax1.set_title(f'Token Embeddings (p={p}) - Fourier Projection', fontsize=13, fontweight='bold', color=text_color)
    ax1.grid(True, alpha=0.2, color=grid_color)
    
    # === Circularity meter ===
    ax_circ = fig.add_axes([0.65, 0.55, 0.12, 0.35])
    ax_circ.set_facecolor('#161b22')
    
    # Vertical bar
    bar_color = plt.cm.RdYlGn(circularity)
    ax_circ.bar([0], [circularity], width=0.6, color=bar_color, edgecolor='white', linewidth=2)
    ax_circ.axhline(y=0.8, color='white', linestyle='--', alpha=0.5)
    ax_circ.set_ylim(0, 1)
    ax_circ.set_xlim(-0.5, 0.5)
    ax_circ.set_xticks([])
    ax_circ.set_ylabel('Circularity', fontsize=11)
    ax_circ.text(0, min(circularity + 0.05, 0.95), f'{circularity:.3f}', ha='center', fontsize=14, 
                fontweight='bold', color=text_color)
    
    # === Fourier spectrum (replaces PCA spectrum) ===
    ax_sv = fig.add_axes([0.82, 0.55, 0.15, 0.35])
    ax_sv.set_facecolor('#161b22')
    
    # Show Fourier frequency strengths
    n_freqs = min(15, len(freq_strengths))
    freq_normalized = freq_strengths[:n_freqs] / (freq_strengths.max() + 1e-10)
    bar_colors = ['#f78166' if i == dominant_freq else '#58a6ff' for i in range(n_freqs)]
    ax_sv.bar(range(n_freqs), freq_normalized, color=bar_colors, alpha=0.8)
    
    ax_sv.set_xlabel('Frequency k', fontsize=10)
    ax_sv.set_ylabel('Strength', fontsize=10)
    ax_sv.set_title(f'Fourier Spectrum\n(k={dominant_freq} explains {freq_explained_var:.1%})', fontsize=10, color=text_color)
    ax_sv.grid(True, alpha=0.2, color=grid_color, axis='y')
    
    # === Training progress ===
    ax2 = fig.add_axes([0.65, 0.15, 0.32, 0.3])
    ax2.set_facecolor('#161b22')
    
    ax2.fill_between(epochs, test_acc, alpha=0.3, color='#7ee787')
    ax2.plot(epochs, test_acc, color='#7ee787', linewidth=2)
    ax2.axvline(x=epoch, color='white', linestyle='--', alpha=0.7)
    ax2.scatter([epoch], [test_acc[-1]], color='white', s=100, zorder=5)
    
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Test Accuracy', fontsize=10)
    ax2.set_xlim(0, history['config']['n_epochs'])
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2, color=grid_color)
    
    # === Title ===
    fig.suptitle(f'Embedding Circle Emergence | Epoch {epoch:,}', 
                fontsize=18, fontweight='bold', color='white', y=0.97)
    
    # Stats text
    stats = f'Circularity: {circularity:.3f} | Freq k={dominant_freq} Var: {freq_explained_var:.1%} | Test Acc: {test_acc[-1]:.1%}'
    fig.text(0.5, 0.02, stats, ha='center', fontsize=11, color='#888888')
    
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return circularity, freq_explained_var


def generate_all_circle_frames(checkpoints_dir, history_path, output_dir, device='cuda'):
    """Generate circle frames for all checkpoints.
    
    First determines the dominant Fourier frequency from the final checkpoint,
    then uses that frequency consistently across all frames for smooth animation.
    """
    
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
    print(f"Output directory: {output_path}")
    
    # Determine dominant frequency from final checkpoint for consistency
    print("Detecting dominant Fourier frequency from final checkpoint...")
    final_cp = checkpoint_files[-1]
    model, config, _ = load_checkpoint(final_cp, device)
    final_embed_data = get_embedding_fourier(model)
    fixed_freq = final_embed_data['dominant_freq']
    print(f"Using fixed frequency k={fixed_freq} for all frames")
    
    history_epochs = history['epochs']
    
    for i, cp_path in enumerate(tqdm(checkpoint_files, desc="Generating circle frames")):
        cp = torch.load(cp_path, map_location='cpu')
        cp_epoch = cp.get('epoch', 0)
        
        frame_idx = min(range(len(history_epochs)), 
                       key=lambda j: abs(history_epochs[j] - cp_epoch))
        
        output_file = output_path / f"frame_{i:05d}.png"
        generate_circle_frame(cp_path, output_file, history, frame_idx, device, fixed_freq=fixed_freq)
    
    print(f"\nFrames saved to {output_path}")
    print(f"\nTo create video:")
    print(f"  ffmpeg -framerate 4 -i {output_path}/frame_%05d.png -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v mpeg4 -q:v 3 circle_emergence.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=str)
    parser.add_argument("history_path", type=str)
    parser.add_argument("--output_dir", type=str, default="frames_circle")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    generate_all_circle_frames(
        args.checkpoints_dir,
        args.history_path,
        args.output_dir,
        args.device
    )
