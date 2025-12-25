"""
Training script for Addition Transformer.

Key features:
1. Variable-length numbers (1-5 digits)
2. Sparse training (small fraction of possible pairs)
3. Curriculum learning (start small, grow to larger numbers)
4. Tracks interpolation and extrapolation accuracy
"""

import warnings
# Suppress ROCm/HIP warnings (optimized kernels not fully available for gfx1151 yet)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with memory efficient attention.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")
warnings.filterwarnings("ignore", message=".*HIPBLAS_STATUS_NOT_SUPPORTED.*")  # hipBLASLt fallback warnings

import torch

# Enable TF32 for better performance on supported GPUs (Ampere+, RDNA3+)
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import json
from pathlib import Path
from collections import defaultdict

from model import AdditionTransformer, create_addition_sample, collate_batch


class AdditionDataset(Dataset):
    """Dataset of addition problems."""
    
    def __init__(
        self,
        n_samples: int,
        max_digits: int = 5,
        min_digits: int = 1,
        seed: int = None
    ):
        self.samples = []
        
        if seed is not None:
            random.seed(seed)
        
        seen = set()
        while len(self.samples) < n_samples:
            # Random digit counts
            digits_a = random.randint(min_digits, max_digits)
            digits_b = random.randint(min_digits, max_digits)
            
            # Generate numbers
            min_a = 10**(digits_a-1) if digits_a > 1 else 0
            max_a = 10**digits_a - 1
            min_b = 10**(digits_b-1) if digits_b > 1 else 0
            max_b = 10**digits_b - 1
            
            a = random.randint(min_a, max_a)
            b = random.randint(min_b, max_b)
            
            # Avoid duplicates
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            
            equation = f"{a}+{b}="
            answer = str(a + b)
            self.samples.append((equation, answer))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_accuracy(model, test_samples, device, verbose=False):
    """
    Evaluate model accuracy on test samples.
    
    Returns:
        accuracy: float
        results: dict with per-digit-length breakdown
    """
    model.eval()
    
    correct = 0
    total = 0
    by_length = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for eq, expected in test_samples:
            src = torch.tensor([model.tokenize(eq)], device=device)
            output = model.generate(src, max_len=12)
            generated = model.detokenize(output[0].tolist())
            
            # Check correctness
            is_correct = (generated == expected)
            if is_correct:
                correct += 1
            total += 1
            
            # Track by answer length
            ans_len = len(expected)
            by_length[ans_len]['total'] += 1
            if is_correct:
                by_length[ans_len]['correct'] += 1
            
            if verbose and not is_correct:
                print(f"  WRONG: {eq} → {generated} (expected {expected})")
    
    accuracy = correct / total if total > 0 else 0
    
    # Convert by_length to accuracy
    results = {
        'overall': accuracy,
        'by_answer_length': {
            k: v['correct'] / v['total'] if v['total'] > 0 else 0
            for k, v in sorted(by_length.items())
        }
    }
    
    return accuracy, results


def train(
    n_epochs: int = 50000,
    batch_size: int = 512,  # Larger batch for better GPU utilization
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    train_samples: int = 50000,
    test_samples: int = 5000,
    max_digits: int = 5,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 512,
    resume: bool = False,
    seed: int = 42,
    compile_model: bool = False
):
    """Train the addition model."""
    
    import sys
    sys.path.insert(0, '..')
    from device_utils import get_device
    device = get_device()
    
    print("=" * 60)
    print("Addition Transformer Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Max digits: {max_digits}")
    print(f"Training samples: {train_samples:,}")
    print(f"Test samples: {test_samples:,}")
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Training: random pairs, curriculum will control digit range
    train_data = AdditionDataset(train_samples, max_digits=max_digits, seed=seed)
    
    # Test sets:
    # 1. Interpolation: same digit range as training, different pairs
    test_interp = AdditionDataset(test_samples // 2, max_digits=max_digits, seed=seed + 1000)
    
    # 2. Extrapolation: 6-digit numbers (beyond training)
    test_extrap = AdditionDataset(test_samples // 2, max_digits=6, min_digits=6, seed=seed + 2000)
    
    print(f"Train: {len(train_data)} samples")
    print(f"Test interpolation: {len(test_interp)} samples (1-{max_digits} digits)")
    print(f"Test extrapolation: {len(test_extrap)} samples (6 digits)")
    
    # Create model
    model = AdditionTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
        d_ff=d_ff
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Optional: compile model with torch.compile for potential speedup
    if compile_model:
        print("Compiling model with torch.compile (first epoch will be slow)...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=model.TOKENS['<pad>'])
    
    # Output directory
    output_path = Path("checkpoints")
    output_path.mkdir(exist_ok=True)
    
    # History
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'interp_acc': [],
        'extrap_acc': [],
        'lr': []
    }
    
    # Resume
    checkpoint_path = output_path / "training_state.pt"
    start_epoch = 0
    
    if resume and checkpoint_path.exists():
        print(f"\nResuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10000, T_mult=2, eta_min=1e-5
    )
    
    # Training loop
    print(f"\nTraining from epoch {start_epoch} to {n_epochs}...")
    pbar = tqdm(range(start_epoch, n_epochs), desc="Training")
    
    train_samples_list = train_data.samples
    
    for epoch in pbar:
        model.train()
        
        # === Curriculum: gradually increase digit complexity ===
        if epoch < 5000:
            # Phase 1: 1-2 digit numbers only
            curriculum_samples = [(eq, ans) for eq, ans in train_samples_list 
                                  if len(ans) <= 3]  # Sum of 2-digit numbers is at most 3 digits
        elif epoch < 15000:
            # Phase 2: up to 3 digits
            curriculum_samples = [(eq, ans) for eq, ans in train_samples_list
                                  if len(ans) <= 4]
        elif epoch < 30000:
            # Phase 3: up to 4 digits
            curriculum_samples = [(eq, ans) for eq, ans in train_samples_list
                                  if len(ans) <= 5]
        else:
            # Phase 4: all
            curriculum_samples = train_samples_list
        
        # Sample batch
        batch_samples = random.sample(curriculum_samples, min(batch_size, len(curriculum_samples)))
        
        # Collate
        src, tgt_in, tgt_out, src_mask = collate_batch(batch_samples, device)
        
        # Forward
        logits = model(src, tgt_in, src_mask)
        
        # Loss (flatten for cross entropy)
        loss = criterion(logits.view(-1, model.VOCAB_SIZE), tgt_out.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # === Logging (every 5000 to match checkpoints; use sweep/compendium scripts for detailed analysis) ===
        if epoch % 5000 == 0:
            model.eval()
            
            # Training accuracy (quick sample)
            train_sample = random.sample(curriculum_samples, min(200, len(curriculum_samples)))
            train_acc, _ = evaluate_accuracy(model, train_sample, device)
            
            # Interpolation accuracy
            interp_sample = random.sample(test_interp.samples, min(500, len(test_interp)))
            interp_acc, _ = evaluate_accuracy(model, interp_sample, device)
            
            # Extrapolation accuracy
            extrap_sample = random.sample(test_extrap.samples, min(500, len(test_extrap)))
            extrap_acc, _ = evaluate_accuracy(model, extrap_sample, device)
            
            current_lr = scheduler.get_last_lr()[0]
            
            history['epoch'].append(epoch)
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['interp_acc'].append(interp_acc)
            history['extrap_acc'].append(extrap_acc)
            history['lr'].append(current_lr)
            
            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'T': f"{train_acc:.0%}",
                'I': f"{interp_acc:.0%}",
                'E': f"{extrap_acc:.0%}"
            })
        
        # === Checkpointing ===
        if epoch % 5000 == 0 and epoch > 0:
            torch.save(model.state_dict(), output_path / f"model_e{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            with open(output_path / "history.json", "w") as f:
                json.dump(history, f, indent=2)
    
    # Final save
    torch.save(model.state_dict(), output_path / "model_final.pt")
    torch.save({
        'epoch': n_epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'history': history
    }, checkpoint_path)
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train accuracy: {history['train_acc'][-1]:.1%}")
    print(f"Final interpolation accuracy: {history['interp_acc'][-1]:.1%}")
    print(f"Final extrapolation accuracy: {history['extrap_acc'][-1]:.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Addition Transformer")
    parser.add_argument("--n_epochs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--test_samples", type=int, default=5000)
    parser.add_argument("--max_digits", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for potential speedup (experimental on AMD)")
    
    args = parser.parse_args()
    
    train(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        max_digits=args.max_digits,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        resume=args.resume,
        seed=args.seed,
        compile_model=args.compile
    )
