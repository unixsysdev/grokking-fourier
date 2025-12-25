"""
Training loop with detailed metrics for phase transition visualization.

Saves checkpoints and metrics frequently for animation generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model import OneLayerTransformer, train_test_split, create_modular_addition_data


def compute_loss_and_accuracy(model, a, b, targets):
    """Compute cross-entropy loss and accuracy."""
    logits = model(a, b)
    loss = F.cross_entropy(logits, targets)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean()
    return loss, accuracy


def compute_fourier_metrics(model, device):
    """
    Compute Fourier-related metrics for the current model state.
    
    Returns dict with:
    - fourier_strength: Average R² of sin/cos fit across neurons
    - key_frequencies: List of dominant frequencies
    - top_freq_strength: Strength of the strongest frequency
    """
    p = model.p
    
    # Get neuron-logit map W_L = W_U @ W_out
    W_L = model.get_neuron_logit_map().detach()  # (p, d_mlp)
    
    # Create Fourier basis
    x = torch.arange(p, device=device, dtype=torch.float32)
    n_freqs = p // 2 + 1
    
    # Compute Fourier coefficients for each neuron
    freq_strengths = []
    for k in range(1, n_freqs):  # Skip DC component
        angle = 2 * np.pi * k * x / p
        cos_basis = torch.cos(angle)
        sin_basis = torch.sin(angle)
        
        # Project W_L onto this frequency
        cos_proj = (W_L.T @ cos_basis) / p  # (d_mlp,)
        sin_proj = (W_L.T @ sin_basis) / p  # (d_mlp,)
        
        # Strength for this frequency across all neurons
        strength = (cos_proj**2 + sin_proj**2).sum().item()
        freq_strengths.append(strength)
    
    freq_strengths = np.array(freq_strengths)
    
    # Normalize by total variance
    total_var = (W_L**2).sum().item()
    if total_var > 0:
        freq_strengths_norm = freq_strengths / total_var
    else:
        freq_strengths_norm = freq_strengths
    
    # Find key frequencies (top 5)
    top_k = min(5, len(freq_strengths))
    key_freq_indices = np.argsort(freq_strengths)[-top_k:][::-1]
    key_frequencies = (key_freq_indices + 1).tolist()  # +1 because we skipped k=0
    
    # Fourier strength = fraction of variance explained by top frequencies
    fourier_strength = freq_strengths_norm[key_freq_indices].sum()
    top_freq_strength = freq_strengths_norm[key_freq_indices[0]] if len(key_freq_indices) > 0 else 0
    
    return {
        'fourier_strength': float(fourier_strength),
        'top_freq_strength': float(top_freq_strength),
        'key_frequencies': key_frequencies,
        'freq_spectrum': freq_strengths_norm.tolist(),
    }


def compute_embedding_geometry(model, device):
    """
    Analyze the geometry of token embeddings.
    
    Returns:
    - circularity: How well embeddings lie on a circle (via PCA)
    - pca_components: First 2 PCA components for visualization
    """
    p = model.p
    
    # Get embeddings for number tokens (exclude = token)
    W_E = model.token_embed.weight[:p, :].detach()  # (p, d_model)
    
    # Center the embeddings
    W_E_centered = W_E - W_E.mean(dim=0, keepdim=True)
    
    # PCA via SVD
    U, S, Vh = torch.linalg.svd(W_E_centered, full_matrices=False)
    
    # Project onto first 2 components
    pca_2d = (W_E_centered @ Vh[:2, :].T).cpu().numpy()  # (p, 2)
    
    # Measure circularity: variance explained by first 2 components
    total_var = (S**2).sum().item()
    var_2d = (S[:2]**2).sum().item()
    circularity = var_2d / total_var if total_var > 0 else 0
    
    # Also compute how well points lie on a circle
    # Normalize to unit circle and measure deviation
    norms = np.linalg.norm(pca_2d, axis=1)
    if norms.mean() > 0:
        circle_fit = 1 - np.std(norms) / np.mean(norms)
    else:
        circle_fit = 0
    
    return {
        'circularity': float(circularity),
        'circle_fit': float(circle_fit),
        'pca_2d': pca_2d.tolist(),
        'singular_values': S[:10].cpu().numpy().tolist(),
    }


def compute_weight_stats(model):
    """Compute statistics about model weights."""
    total_norm = 0
    total_params = 0
    
    for param in model.parameters():
        total_norm += (param**2).sum().item()
        total_params += param.numel()
    
    return {
        'weight_norm': float(np.sqrt(total_norm)),
        'weight_norm_squared': float(total_norm),
        'n_params': total_params,
    }


def train_with_metrics(
    p: int = 113,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    train_frac: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    n_epochs: int = 30000,
    log_every: int = 10,
    checkpoint_every: int = 100,
    compute_metrics_every: int = 100,
    seed: int = 42,
    output_dir: str = "checkpoints",
    run_name: str = "run",
):
    """
    Train with detailed metrics for visualization.
    
    Saves:
    - Checkpoints every `checkpoint_every` epochs
    - Detailed metrics every `compute_metrics_every` epochs
    - Full history for animation generation
    """
    # Setup
    sys.path.append(str(Path(__file__).parent.parent))
    from device_utils import get_device
    device = get_device()
    
    print(f"=" * 60)
    print(f"Phase Transition Training: {run_name}")
    print(f"=" * 60)
    print(f"p={p}, train_frac={train_frac}, weight_decay={weight_decay}")
    print(f"Device: {device}")
    
    torch.manual_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = OneLayerTransformer(
        p=p,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
    ).to(device)
    
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create data
    train_data, test_data = train_test_split(p, train_frac, device, seed)
    a_train, b_train, targets_train = train_data
    a_test, b_test, targets_test = test_data
    
    print(f"Train samples: {len(a_train)}, Test samples: {len(a_test)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # History for animation
    history = {
        'config': {
            'p': p,
            'd_model': d_model,
            'n_heads': n_heads,
            'd_mlp': d_mlp,
            'train_frac': train_frac,
            'lr': lr,
            'weight_decay': weight_decay,
            'n_epochs': n_epochs,
            'seed': seed,
        },
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'fourier_strength': [],
        'top_freq_strength': [],
        'key_frequencies': [],
        'freq_spectrum': [],
        'weight_norm': [],
        'circularity': [],
        'circle_fit': [],
        'pca_2d': [],
    }
    
    # Training loop
    pbar = tqdm(range(n_epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        
        # Forward + backward
        optimizer.zero_grad()
        train_loss, train_acc = compute_loss_and_accuracy(
            model, a_train, b_train, targets_train
        )
        train_loss.backward()
        optimizer.step()
        
        # Log basic metrics
        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = compute_loss_and_accuracy(
                    model, a_test, b_test, targets_test
                )
            
            pbar.set_postfix({
                'tr_loss': f"{train_loss.item():.4f}",
                'te_loss': f"{test_loss.item():.4f}",
                'tr_acc': f"{train_acc.item():.3f}",
                'te_acc': f"{test_acc.item():.3f}",
            })
        
        # Compute detailed metrics
        if epoch % compute_metrics_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = compute_loss_and_accuracy(
                    model, a_test, b_test, targets_test
                )
                
                # Fourier metrics
                fourier_metrics = compute_fourier_metrics(model, device)
                
                # Embedding geometry
                embed_metrics = compute_embedding_geometry(model, device)
                
                # Weight stats
                weight_stats = compute_weight_stats(model)
            
            # Store in history
            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
            history['train_acc'].append(train_acc.item())
            history['test_acc'].append(test_acc.item())
            history['fourier_strength'].append(fourier_metrics['fourier_strength'])
            history['top_freq_strength'].append(fourier_metrics['top_freq_strength'])
            history['key_frequencies'].append(fourier_metrics['key_frequencies'])
            history['freq_spectrum'].append(fourier_metrics['freq_spectrum'])
            history['weight_norm'].append(weight_stats['weight_norm'])
            history['circularity'].append(embed_metrics['circularity'])
            history['circle_fit'].append(embed_metrics['circle_fit'])
            history['pca_2d'].append(embed_metrics['pca_2d'])
        
        # Save checkpoint
        if epoch % checkpoint_every == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item() if 'test_loss' in dir() else None,
                'train_acc': train_acc.item(),
                'test_acc': test_acc.item() if 'test_acc' in dir() else None,
            }
            torch.save(checkpoint, output_path / f"checkpoint_e{epoch:06d}.pt")
    
    # Save final checkpoint and history
    final_checkpoint = {
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, output_path / "checkpoint_final.pt")
    
    # Save history
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f)
    
    print(f"\nTraining complete!")
    print(f"Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"Final test acc: {history['test_acc'][-1]:.4f}")
    print(f"Final Fourier strength: {history['fourier_strength'][-1]:.4f}")
    print(f"Results saved to: {output_path}")
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with metrics for phase transition visualization")
    parser.add_argument("--p", type=int, default=113, help="Prime modulus")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=512)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--compute_metrics_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="phase_transition/checkpoints")
    parser.add_argument("--run_name", type=str, default="default")
    
    args = parser.parse_args()
    
    train_with_metrics(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        train_frac=args.train_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        compute_metrics_every=args.compute_metrics_every,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
