"""
Sparse Universal Training with Improved SinPE

Key differences from train_ce_universal.py:
1. Uses ImprovedSinusoidalModulusEncoding (log-scaled, higher frequencies)
2. Sparse training: ~30 moduli spread across full range 2-199
3. Tests interpolation on ALL moduli not in training set
4. Goal: prove true generalization by training on sparse points
"""

import torch
import torch.nn as nn
from model_sparse import SparseUniversalTransformer
from tqdm import tqdm
import random
import json
from pathlib import Path


def generate_sparse_data(moduli, samples_per_mod=100):
    """Generate training data for arbitrary moduli."""
    all_data = []
    for m in moduli:
        if m < 20:
            # Exhaustive for small moduli
            for a in range(m):
                for b in range(m):
                    all_data.append((m, a, b, (a + b) % m))
        else:
            # Sample for larger moduli
            for _ in range(samples_per_mod):
                a = random.randint(0, m - 1)
                b = random.randint(0, m - 1)
                all_data.append((m, a, b, (a + b) % m))
    random.shuffle(all_data)
    return all_data


def train_ce_sparse(resume: bool = False, checkpoint: str = None, start_epoch: int = 0, n_epochs: int = 150000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training SPARSE with Improved SinPE on: {device} ---")
    
    # SPARSE training set: ~30 moduli spread across full range
    # Key insight: if model truly learns the algorithm, it should interpolate EVERYTHING
    train_moduli = [
        # Small (dense coverage for learning basics)
        2, 3, 5, 7, 11, 13,
        # Medium (sparse)
        19, 28, 37, 46, 55, 64, 73, 82, 91,
        # Large (sparse, into extrapolation territory)
        100, 112, 125, 138, 150, 163, 175, 188, 199
    ]
    
    # Test on EVERYTHING else - true interpolation test
    all_moduli = set(range(2, 200))
    test_moduli_interpolate = sorted(all_moduli - set(train_moduli))
    
    print(f"Training on {len(train_moduli)} sparse moduli: {train_moduli}")
    print(f"Testing interpolation on {len(test_moduli_interpolate)} held-out moduli")
    
    # Same model size as universal for fair comparison
    model = SparseUniversalTransformer(
        max_p=250,      # Support up to 249
        d_model=256,    
        n_heads=8,      
        d_mlp=1024,     
        d_mem=256       
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    output_path = Path("checkpoints/ce_sparse")
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = {"epoch": [], "train_loss": [], "train_acc": [], "interp_acc": []}
    
    # Resume from full training state
    checkpoint_path = output_path / "training_state.pt"
    if resume and checkpoint_path.exists():
        print(f"Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")
    elif resume:
        print("No training_state.pt found, starting fresh...")
    
    # Load model weights only
    if checkpoint:
        print(f"Loading model weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        history_path = output_path / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            print(f"Loaded history with {len(history['epoch'])} entries")
        print(f"Starting from epoch {start_epoch} with fresh optimizer")
    
    # Scheduler
    remaining_epochs = n_epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    print(f"Training from epoch {start_epoch} to {n_epochs} ({remaining_epochs} epochs, LR: {scheduler.get_last_lr()[0]:.2e})")
    
    batch_size = 1024
    pbar = tqdm(range(start_epoch, n_epochs), desc="Sparse CE Training")
    
    for epoch in pbar:
        model.train()
        
        # No curriculum - use all training moduli from start
        # (sparse set is already small, curriculum not needed)
        batch = generate_sparse_data(train_moduli, samples_per_mod=100)
        samples = random.sample(batch, min(len(batch), batch_size))
        
        m_v = torch.tensor([x[0] for x in samples], device=device)
        a_v = torch.tensor([x[1] for x in samples], device=device)
        b_v = torch.tensor([x[2] for x in samples], device=device)
        t_v = torch.tensor([x[3] for x in samples], device=device)
        
        logits = model(m_v, a_v, b_v)
        loss = criterion(logits, t_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                # Training accuracy
                train_preds = logits.argmax(dim=-1)
                train_acc = (train_preds == t_v).float().mean().item()
                
                # Test interpolation (sample from held-out moduli)
                test_sample_moduli = random.sample(test_moduli_interpolate, min(30, len(test_moduli_interpolate)))
                interp_data = generate_sparse_data(test_sample_moduli, samples_per_mod=50)
                interp_samples = random.sample(interp_data, min(len(interp_data), batch_size))
                
                m_i = torch.tensor([x[0] for x in interp_samples], device=device)
                a_i = torch.tensor([x[1] for x in interp_samples], device=device)
                b_i = torch.tensor([x[2] for x in interp_samples], device=device)
                t_i = torch.tensor([x[3] for x in interp_samples], device=device)
                
                interp_logits = model(m_i, a_i, b_i)
                interp_acc = (interp_logits.argmax(dim=-1) == t_i).float().mean().item()
                
                history["epoch"].append(epoch)
                history["train_loss"].append(loss.item())
                history["train_acc"].append(train_acc)
                history["interp_acc"].append(interp_acc)
                
                pbar.set_postfix({
                    "L": f"{loss.item():.4f}",
                    "T": f"{train_acc:.1%}",
                    "I": f"{interp_acc:.1%}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}"
                })
                
                if epoch % 5000 == 0:
                    torch.save(model.state_dict(), output_path / f"sparse_e{epoch}.pt")
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    }, checkpoint_path)
                    with open(output_path / "history.json", "w") as f:
                        json.dump(history, f)

    torch.save(model.state_dict(), output_path / "sparse_final.pt")
    torch.save({
        "epoch": n_epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
    }, checkpoint_path)
    
    print(f"\nTraining complete! Final results:")
    print(f"  Training accuracy: {history['train_acc'][-1]:.1%}")
    print(f"  Interpolation accuracy: {history['interp_acc'][-1]:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Sparse CE model with improved SinPE")
    parser.add_argument("--resume", action="store_true", help="Resume from training_state.pt")
    parser.add_argument("--checkpoint", type=str, help="Load model weights from a .pt file")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch")
    parser.add_argument("--n_epochs", type=int, default=150000, help="Total epochs to train")
    args = parser.parse_args()
    train_ce_sparse(resume=args.resume, checkpoint=args.checkpoint, start_epoch=args.start_epoch, n_epochs=args.n_epochs)
