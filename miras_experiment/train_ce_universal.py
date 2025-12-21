"""
Universal Modular Addition Training (CE)

Key differences from train_ce.py:
- Trains on ALL moduli from 2-149 (not just primes)
- Includes larger moduli in training (up to 120)
- Larger model capacity (d_model=256, d_mem=256, d_mlp=1024)
- Tests on held-out moduli to verify true generalization
"""

import torch
import torch.nn as nn
from model_miras import UniversalFourierTransformer
from tqdm import tqdm
import random
import json
from pathlib import Path


def generate_universal_data(moduli, samples_per_mod=100):
    """Generate training data for arbitrary moduli (not just primes)."""
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


def train_ce_universal(resume: bool = False, checkpoint: str = None, start_epoch: int = 0, n_epochs: int = 100000):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training UNIVERSAL CE (all moduli, larger model) on: {device} ---")
    
    # Use ALL moduli from 2-120 for training (not just primes!)
    # This forces the model to learn the continuous relationship
    all_train_moduli = list(range(2, 121))  # 2 to 120 inclusive
    
    # Hold out some moduli for testing generalization
    # Mix of: some within training range (interpolation) and beyond (extrapolation)
    test_moduli_interpolate = [15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]  # within range but held out
    test_moduli_extrapolate = [121, 127, 131, 137, 139, 143, 149]  # beyond training range
    
    # Remove interpolation test moduli from training
    train_moduli = [m for m in all_train_moduli if m not in test_moduli_interpolate]
    
    print(f"Training moduli: {len(train_moduli)} values from 2-120 (excluding held-out)")
    print(f"Test (interpolate): {test_moduli_interpolate}")
    print(f"Test (extrapolate): {test_moduli_extrapolate}")
    
    # Larger model for more capacity
    model = UniversalFourierTransformer(
        max_p=200,      # Support up to 199
        d_model=256,    # 2x original
        n_heads=8,      # 2x original
        d_mlp=1024,     # 2x original
        d_mem=256       # 2x original
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    output_path = Path("checkpoints/ce_universal")
    output_path.mkdir(parents=True, exist_ok=True)
    
    history = {"epoch": [], "train_loss": [], "interp_acc": [], "extrap_acc": []}
    
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
    
    # Load model weights only (for continuing from old checkpoints)
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
    
    # Create scheduler
    remaining_epochs = n_epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    print(f"Training from epoch {start_epoch} to {n_epochs} ({remaining_epochs} epochs, LR: {scheduler.get_last_lr()[0]:.2e})")
    
    batch_size = 1024
    pbar = tqdm(range(start_epoch, n_epochs), desc="Universal CE Training")
    
    for epoch in pbar:
        model.train()
        
        # Curriculum: start with smaller moduli, gradually include larger ones
        if epoch < 5000:
            current_moduli = [m for m in train_moduli if m <= 30]
        elif epoch < 15000:
            current_moduli = [m for m in train_moduli if m <= 60]
        elif epoch < 30000:
            current_moduli = [m for m in train_moduli if m <= 90]
        else:
            current_moduli = train_moduli
        
        batch = generate_universal_data(current_moduli, samples_per_mod=50)
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
                # Test interpolation (held-out moduli within training range)
                interp_data = generate_universal_data(test_moduli_interpolate, samples_per_mod=100)
                interp_samples = random.sample(interp_data, min(len(interp_data), batch_size))
                m_i = torch.tensor([x[0] for x in interp_samples], device=device)
                a_i = torch.tensor([x[1] for x in interp_samples], device=device)
                b_i = torch.tensor([x[2] for x in interp_samples], device=device)
                t_i = torch.tensor([x[3] for x in interp_samples], device=device)
                interp_acc = (model(m_i, a_i, b_i).argmax(dim=-1) == t_i).float().mean().item()
                
                # Test extrapolation (moduli beyond training range)
                extrap_data = generate_universal_data(test_moduli_extrapolate, samples_per_mod=100)
                extrap_samples = random.sample(extrap_data, min(len(extrap_data), batch_size))
                m_e = torch.tensor([x[0] for x in extrap_samples], device=device)
                a_e = torch.tensor([x[1] for x in extrap_samples], device=device)
                b_e = torch.tensor([x[2] for x in extrap_samples], device=device)
                t_e = torch.tensor([x[3] for x in extrap_samples], device=device)
                extrap_acc = (model(m_e, a_e, b_e).argmax(dim=-1) == t_e).float().mean().item()
                
                history["epoch"].append(epoch)
                history["train_loss"].append(loss.item())
                history["interp_acc"].append(interp_acc)
                history["extrap_acc"].append(extrap_acc)
                pbar.set_postfix({
                    "L": f"{loss.item():.4f}",
                    "I": f"{interp_acc:.1%}",
                    "E": f"{extrap_acc:.1%}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}"
                })
                
                if epoch % 5000 == 0:
                    torch.save(model.state_dict(), output_path / f"universal_e{epoch}.pt")
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    }, checkpoint_path)
                    with open(output_path / "history.json", "w") as f:
                        json.dump(history, f)

    torch.save(model.state_dict(), output_path / "universal_final.pt")
    torch.save({
        "epoch": n_epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
    }, checkpoint_path)
    
    print(f"\nTraining complete! Final results:")
    print(f"  Interpolation accuracy: {history['interp_acc'][-1]:.1%}")
    print(f"  Extrapolation accuracy: {history['extrap_acc'][-1]:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Universal CE model (all moduli, larger capacity)")
    parser.add_argument("--resume", action="store_true", help="Resume from training_state.pt")
    parser.add_argument("--checkpoint", type=str, help="Load model weights from a .pt file")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch (use with --checkpoint)")
    parser.add_argument("--n_epochs", type=int, default=100000, help="Total epochs to train")
    args = parser.parse_args()
    train_ce_universal(resume=args.resume, checkpoint=args.checkpoint, start_epoch=args.start_epoch, n_epochs=args.n_epochs)
