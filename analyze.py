"""
Fourier analysis of trained modular addition transformer.

This script replicates the key analyses from the grokking paper:
1. Fourier transform of embedding matrix W_E
2. Fourier transform of neuron-logit map W_L
3. Periodicity in attention patterns
4. Periodicity in neuron activations
5. 2D Fourier transform of logits
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import OneLayerTransformer, create_modular_addition_data


def discrete_fourier_basis(p: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create the discrete Fourier basis for a prime p.
    
    Returns:
        cos_basis: (p, p//2 + 1) - cosine components
        sin_basis: (p, p//2 + 1) - sine components
    
    For frequency k, the basis functions are:
        cos_k(x) = cos(2πkx/p)
        sin_k(x) = sin(2πkx/p)
    """
    x = torch.arange(p, device=device, dtype=torch.float32)
    k = torch.arange(p // 2 + 1, device=device, dtype=torch.float32)
    
    # Compute 2πkx/p for all x, k combinations
    angles = 2 * np.pi * torch.outer(x, k) / p  # (p, p//2+1)
    
    cos_basis = torch.cos(angles)
    sin_basis = torch.sin(angles)
    
    return cos_basis, sin_basis


def fourier_transform_1d(matrix: torch.Tensor, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Fourier transform along the first axis of a matrix.
    
    Args:
        matrix: (p, d) tensor where we take Fourier transform along dim 0
        p: Prime modulus (should match matrix.shape[0])
        
    Returns:
        cos_coeffs: (p//2+1, d) - cosine Fourier coefficients
        sin_coeffs: (p//2+1, d) - sine Fourier coefficients
    """
    device = matrix.device
    cos_basis, sin_basis = discrete_fourier_basis(p, device)
    
    # Normalize
    cos_basis = cos_basis / np.sqrt(p / 2)
    sin_basis = sin_basis / np.sqrt(p / 2)
    
    # Project onto Fourier basis
    cos_coeffs = cos_basis.T @ matrix  # (p//2+1, d)
    sin_coeffs = sin_basis.T @ matrix  # (p//2+1, d)
    
    return cos_coeffs, sin_coeffs


def analyze_embedding_matrix(model: OneLayerTransformer, save_dir: Path):
    """
    Analyze the Fourier structure of the embedding matrix W_E.
    
    The paper shows that W_E is sparse in the Fourier basis,
    with only a few "key frequencies" having significant norm.
    """
    p = model.p
    device = next(model.parameters()).device
    
    # Get embedding matrix for the p number tokens (not the = token)
    W_E = model.token_embed.weight[:p, :].detach()  # (p, d_model)
    
    # Fourier transform
    cos_coeffs, sin_coeffs = fourier_transform_1d(W_E, p)
    
    # Compute norm of each frequency component
    cos_norms = torch.norm(cos_coeffs, dim=1).cpu().numpy()
    sin_norms = torch.norm(sin_coeffs, dim=1).cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_freqs = len(cos_norms)
    x = np.arange(n_freqs)
    width = 0.35
    
    ax.bar(x - width/2, cos_norms, width, label='cos', alpha=0.8)
    ax.bar(x + width/2, sin_norms, width, label='sin', alpha=0.8)
    
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Norm of Fourier Component')
    ax.set_title('Fourier Components of Embedding Matrix W_E')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'embedding_fourier.png', dpi=150)
    plt.close()
    
    # Find key frequencies (those with significant norm)
    total_norm = np.sqrt(cos_norms**2 + sin_norms**2)
    threshold = total_norm.max() * 0.1
    key_freqs = np.where(total_norm > threshold)[0]
    
    print(f"Embedding matrix key frequencies: {key_freqs}")
    print(f"Their norms: {total_norm[key_freqs]}")
    
    return key_freqs, cos_norms, sin_norms


def analyze_neuron_logit_map(model: OneLayerTransformer, save_dir: Path):
    """
    Analyze the Fourier structure of W_L = W_U @ W_out.
    
    The paper shows that W_L is approximately rank 10, with each direction
    corresponding to sin or cos of one of ~5 key frequencies.
    """
    p = model.p
    
    W_L = model.get_neuron_logit_map().detach()  # (p, d_mlp)
    
    # Fourier transform along the logit (output) dimension
    cos_coeffs, sin_coeffs = fourier_transform_1d(W_L, p)
    
    # Compute norm over the neuron dimension for each frequency
    cos_norms = torch.norm(cos_coeffs, dim=1).cpu().numpy()
    sin_norms = torch.norm(sin_coeffs, dim=1).cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_freqs = len(cos_norms)
    x = np.arange(n_freqs)
    width = 0.35
    
    ax.bar(x - width/2, cos_norms, width, label='cos', alpha=0.8)
    ax.bar(x + width/2, sin_norms, width, label='sin', alpha=0.8)
    
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Norm of Fourier Component')
    ax.set_title('Fourier Components of Neuron-Logit Map W_L')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'neuron_logit_fourier.png', dpi=150)
    plt.close()
    
    # Find key frequencies
    total_norm = np.sqrt(cos_norms**2 + sin_norms**2)
    threshold = total_norm.max() * 0.1
    key_freqs = np.where(total_norm > threshold)[0]
    
    print(f"Neuron-logit map key frequencies: {key_freqs}")
    print(f"Their norms: {total_norm[key_freqs]}")
    
    return key_freqs, cos_norms, sin_norms


def analyze_neuron_activations(model: OneLayerTransformer, save_dir: Path):
    """
    Analyze periodicity in MLP neuron activations.
    
    For each pair (a, b), we compute the activations and look for
    periodic structure. The paper shows that individual neurons
    are well-approximated by degree-2 polynomials of sines/cosines.
    """
    p = model.p
    device = next(model.parameters()).device
    
    # Create all input pairs
    a, b, targets = create_modular_addition_data(p, device)
    
    # Forward pass to get activations
    model.eval()
    with torch.no_grad():
        _ = model(a, b)
        mlp_acts = model.cache['mlp_activations'][:, 2, :]  # (p*p, d_mlp), position 2 is '='
    
    # Reshape to (p, p, d_mlp)
    mlp_acts = mlp_acts.view(p, p, -1)
    
    # Plot a few example neurons
    n_examples = 4
    fig, axes = plt.subplots(1, n_examples, figsize=(16, 4))
    
    for i, ax in enumerate(axes):
        neuron_acts = mlp_acts[:, :, i].cpu().numpy()
        im = ax.imshow(neuron_acts, origin='lower', cmap='viridis')
        ax.set_xlabel('b')
        ax.set_ylabel('a')
        ax.set_title(f'Neuron {i}')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('MLP Neuron Activations (showing periodicity)')
    plt.tight_layout()
    plt.savefig(save_dir / 'neuron_activations.png', dpi=150)
    plt.close()
    
    return mlp_acts


def analyze_logits_2d_fourier(model: OneLayerTransformer, save_dir: Path):
    """
    Compute 2D Fourier transform of logits over inputs (a, b).
    
    The paper shows that the logits have only ~20 significant components,
    corresponding to products of sines/cosines of ~5 key frequencies.
    """
    p = model.p
    device = next(model.parameters()).device
    
    # Create all input pairs
    a, b, targets = create_modular_addition_data(p, device)
    
    # Get logits
    model.eval()
    with torch.no_grad():
        logits = model(a, b)  # (p*p, p)
    
    # Reshape to (p, p, p) - indexed by (a, b, output_c)
    logits = logits.view(p, p, p)
    
    # 2D Fourier transform over (a, b) for each output c
    # Using torch's FFT
    logits_fft = torch.fft.fft2(logits, dim=(0, 1))
    
    # Take norm over the output dimension
    logits_fft_norm = torch.abs(logits_fft).norm(dim=2).cpu().numpy()
    
    # Only show first half (symmetric)
    n_show = p // 2 + 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(logits_fft_norm[:n_show, :n_show], origin='lower', cmap='hot')
    ax.set_xlabel('Frequency in b')
    ax.set_ylabel('Frequency in a')
    ax.set_title('2D Fourier Transform of Logits (norm over output c)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'logits_2d_fourier.png', dpi=150)
    plt.close()
    
    # Find the key frequency combinations
    threshold = logits_fft_norm.max() * 0.1
    key_locs = np.where(logits_fft_norm > threshold)
    print(f"Number of significant 2D Fourier components: {len(key_locs[0])}")
    
    return logits_fft_norm


def analyze_attention_patterns(model: OneLayerTransformer, save_dir: Path):
    """
    Analyze periodicity in attention patterns.
    
    The paper shows that each attention head attends with a periodic pattern
    that corresponds to a specific frequency.
    """
    p = model.p
    device = next(model.parameters()).device
    
    # Create all input pairs
    a, b, targets = create_modular_addition_data(p, device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(a, b)
        attn = model.cache['attn_pattern']  # (batch, heads, seq, seq)
    
    # Get attention from '=' (position 2) to 'a' (position 0)
    # Shape: (p*p, n_heads)
    attn_to_a = attn[:, :, 2, 0]
    
    # Reshape to (p, p, n_heads)
    attn_to_a = attn_to_a.view(p, p, -1)
    
    n_heads = model.n_heads
    fig, axes = plt.subplots(1, n_heads, figsize=(4*n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        pattern = attn_to_a[:, :, i].cpu().numpy()
        im = ax.imshow(pattern, origin='lower', cmap='viridis')
        ax.set_xlabel('b')
        ax.set_ylabel('a')
        ax.set_title(f'Head {i}: attn("=" → "a")')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Attention Patterns (from = to a)')
    plt.tight_layout()
    plt.savefig(save_dir / 'attention_patterns.png', dpi=150)
    plt.close()
    
    return attn_to_a


def plot_training_curves(history: dict, save_dir: Path):
    """Plot training curves (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epoch']
    
    # Loss (log scale)
    ax1.semilogy(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
    ax1.semilogy(epochs, history['test_loss'], label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', alpha=0.8)
    ax2.plot(epochs, history['test_acc'], label='Test Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()


def run_full_analysis(checkpoint_path: str, output_dir: str = "analysis"):
    """
    Run all Fourier analyses on a trained model.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    print(f"Loaded model trained for {checkpoint['epoch']} epochs")
    print(f"Config: {config}")
    
    # Create model and load weights
    from device_utils import get_device
    device = get_device()
    
    model = OneLayerTransformer(
        p=config['p'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_mlp=config['d_mlp'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    save_dir = Path(output_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(checkpoint['history'], save_dir)
    
    # Run analyses
    print("\nAnalyzing embedding matrix...")
    embed_key_freqs, _, _ = analyze_embedding_matrix(model, save_dir)
    
    print("\nAnalyzing neuron-logit map...")
    wl_key_freqs, _, _ = analyze_neuron_logit_map(model, save_dir)
    
    print("\nAnalyzing neuron activations...")
    _ = analyze_neuron_activations(model, save_dir)
    
    print("\nAnalyzing attention patterns...")
    _ = analyze_attention_patterns(model, save_dir)
    
    print("\nAnalyzing 2D Fourier structure of logits...")
    _ = analyze_logits_2d_fourier(model, save_dir)
    
    print(f"\nAnalysis complete! Results saved to {save_dir}/")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="analysis")
    
    args = parser.parse_args()
    
    run_full_analysis(args.checkpoint, args.output_dir)
