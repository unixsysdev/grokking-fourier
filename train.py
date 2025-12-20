"""
Training loop for modular addition transformer.

This replicates the training setup from the grokking paper:
- Full batch gradient descent
- AdamW optimizer with weight decay
- Track train/test loss and accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
from pathlib import Path

from model import OneLayerTransformer, train_test_split


def compute_loss_and_accuracy(model, a, b, targets):
    """Compute cross-entropy loss and accuracy."""
    logits = model(a, b)
    loss = F.cross_entropy(logits, targets)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean()
    return loss, accuracy


def train(
    p: int = 53,
    d_model: int = 128,
    n_heads: int = 4,
    d_mlp: int = 512,
    train_frac: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    n_epochs: int = 20000,
    log_every: int = 100,
    save_every: int = 1000,
    seed: int = 42,
    output_dir: str = "checkpoints",
):
    """
    Train the transformer on modular addition.
    
    Args:
        p: Prime modulus (paper uses 113, we use smaller for speed)
        d_model: Model dimension
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
        train_frac: Fraction of data for training
        lr: Learning rate
        weight_decay: Weight decay (crucial for grokking!)
        n_epochs: Number of training epochs
        log_every: Log metrics every N epochs
        save_every: Save checkpoint every N epochs
        seed: Random seed
        output_dir: Directory for checkpoints
    """
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training modular addition mod {p}")
    print(f"Train fraction: {train_frac}, Total pairs: {p*p}")
    
    torch.manual_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create model
    model = OneLayerTransformer(
        p=p,
        d_model=d_model,
        n_heads=n_heads,
        d_mlp=d_mlp,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create data
    train_data, test_data = train_test_split(p, train_frac, device, seed)
    a_train, b_train, targets_train = train_data
    a_test, b_test, targets_test = test_data
    
    print(f"Train samples: {len(a_train)}, Test samples: {len(a_test)}")
    
    # Optimizer (AdamW with weight decay - crucial!)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    
    # Training loop
    pbar = tqdm(range(n_epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        
        # Full batch training
        optimizer.zero_grad()
        train_loss, train_acc = compute_loss_and_accuracy(
            model, a_train, b_train, targets_train
        )
        train_loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = compute_loss_and_accuracy(
                    model, a_test, b_test, targets_test
                )
            
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
            history['train_acc'].append(train_acc.item())
            history['test_acc'].append(test_acc.item())
            
            pbar.set_postfix({
                'train_loss': f"{train_loss.item():.4f}",
                'test_loss': f"{test_loss.item():.4f}",
                'train_acc': f"{train_acc.item():.3f}",
                'test_acc': f"{test_acc.item():.3f}",
            })
        
        # Save checkpoints
        if epoch % save_every == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'p': p,
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'd_mlp': d_mlp,
                    'train_frac': train_frac,
                    'lr': lr,
                    'weight_decay': weight_decay,
                }
            }
            torch.save(checkpoint, output_path / f"checkpoint_{epoch:06d}.pt")
    
    # Save final model and history
    final_checkpoint = {
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'p': p,
            'd_model': d_model,
            'n_heads': n_heads,
            'd_mlp': d_mlp,
            'train_frac': train_frac,
            'lr': lr,
            'weight_decay': weight_decay,
        }
    }
    torch.save(final_checkpoint, output_path / "checkpoint_final.pt")
    
    # Save history as JSON for easy plotting
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f)
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss: {history['test_loss'][-1]:.6f}")
    print(f"Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"Final test acc: {history['test_acc'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=71, help="Prime modulus")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=512)
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    train(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        train_frac=args.train_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        output_dir=args.output_dir,
    )
