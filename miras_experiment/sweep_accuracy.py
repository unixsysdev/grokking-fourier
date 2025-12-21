import torch
from model_miras import UniversalFourierTransformer
from model_sparse import SparseUniversalTransformer
from pathlib import Path
import json

def get_accuracy(model, p, device):
    model.eval()
    with torch.no_grad():
        a_vals = torch.arange(p, device=device).repeat_interleave(p)
        b_vals = torch.arange(p, device=device).repeat(p)
        p_vals = torch.full_like(a_vals, p)
        
        logits = model(p_vals, a_vals, b_vals)
        acc = (logits.argmax(dim=-1) == (a_vals + b_vals) % p).float().mean().item()
    return acc

def sweep_all(mode="ce", epoch="40000", universal=False, sparse=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if sparse:
        base_check_dir = Path("checkpoints/ce_sparse")
        path_options = [
            base_check_dir / f"sparse_{epoch}.pt",
            base_check_dir / f"sparse_e{epoch}.pt"
        ]
        model = SparseUniversalTransformer(max_p=250, d_model=256, n_heads=8, d_mlp=1024, d_mem=256).to(device)
        test_moduli = list(range(2, 200))
    elif universal:
        base_check_dir = Path("checkpoints/ce_universal")
        path_options = [
            base_check_dir / f"universal_{epoch}.pt",
            base_check_dir / f"universal_e{epoch}.pt"
        ]
        model = UniversalFourierTransformer(max_p=200, d_model=256, n_heads=8, d_mlp=1024, d_mem=256).to(device)
        test_moduli = list(range(2, 200))
    else:
        base_check_dir = Path(f"checkpoints/{mode}_sinpe")
        path_options = [
            base_check_dir / f"{mode}_{epoch}.pt",
            base_check_dir / f"{mode}_e{epoch}.pt"
        ]
        model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
        test_moduli = [p for p in range(2, 150) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    model_path = None
    for p in path_options:
        if p.exists():
            model_path = p
            break
            
    if model_path is None:
        print(f"Error: Could not find checkpoint for {mode} {epoch} in {base_check_dir}")
        return
    
    print(f"Loading {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    accuracies = {}
    print(f"Testing {len(test_moduli)} moduli...")
    for m in test_moduli:
        acc = get_accuracy(model, m, device)
        accuracies[m] = acc
    
    if sparse:
        label = "sparse"
        # Highlight which were training moduli
        train_moduli = {2, 3, 5, 7, 11, 13, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 112, 125, 138, 150, 163, 175, 188, 199}
    else:
        label = "universal" if universal else mode
        train_moduli = set()
    
    print(f"--- Accuracy Sweep for {label} {epoch} ---")
    sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    for m, acc in sorted_acc:
        if acc > 0.05 or m > 120:
            marker = " [TRAIN]" if m in train_moduli else ""
            print(f"m={m:3d}: {acc:6.1%}{marker}")

def find_latest_checkpoint(mode):
    """Find the latest checkpoint for a given mode."""
    base_dir = Path(f"checkpoints/{mode}_sinpe")
    if not base_dir.exists():
        return None
    
    # Look for ce_e*.pt or rl_e*.pt files
    checkpoints = list(base_dir.glob(f"{mode}_e*.pt"))
    if base_dir.glob(f"{mode}_final.pt"):
        checkpoints.extend(base_dir.glob(f"{mode}_final.pt"))
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find max
    def get_epoch(p):
        name = p.stem
        if "final" in name:
            return float('inf')  # final is always latest
        try:
            return int(name.split('e')[-1])
        except:
            return -1
    
    latest = max(checkpoints, key=get_epoch)
    return latest

def find_latest_checkpoint(mode, universal=False, sparse=False):
    """Find the latest checkpoint for a given mode."""
    if sparse:
        base_dir = Path("checkpoints/ce_sparse")
        prefix = "sparse"
    elif universal:
        base_dir = Path("checkpoints/ce_universal")
        prefix = "universal"
    else:
        base_dir = Path(f"checkpoints/{mode}_sinpe")
        prefix = mode
        
    if not base_dir.exists():
        return None
    
    checkpoints = list(base_dir.glob(f"{prefix}_e*.pt"))
    final = list(base_dir.glob(f"{prefix}_final.pt"))
    if final:
        checkpoints.extend(final)
    
    if not checkpoints:
        return None
    
    def get_epoch(p):
        name = p.stem
        if "final" in name:
            return float('inf')
        try:
            return int(name.split('e')[-1])
        except:
            return -1
    
    return max(checkpoints, key=get_epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sweep accuracy across moduli for a checkpoint")
    parser.add_argument("--mode", type=str, choices=["ce", "rl", "universal", "sparse", "both", "all"], default="both",
                        help="Which model to test (ce, rl, universal, sparse, both, or all)")
    parser.add_argument("--epoch", type=str, default="latest",
                        help="Epoch to test (e.g., '40000', 'final', or 'latest')")
    args = parser.parse_args()
    
    if args.mode == "all":
        modes = [("ce", False, False), ("rl", False, False), ("universal", True, False), ("sparse", False, True)]
    elif args.mode == "both":
        modes = [("ce", False, False), ("rl", False, False)]
    elif args.mode == "universal":
        modes = [("universal", True, False)]
    elif args.mode == "sparse":
        modes = [("sparse", False, True)]
    else:
        modes = [(args.mode, False, False)]
    
    for mode, is_universal, is_sparse in modes:
        if args.epoch == "latest":
            checkpoint = find_latest_checkpoint(mode, universal=is_universal, sparse=is_sparse)
            if checkpoint:
                epoch = checkpoint.stem.replace(f"{mode}_", "").replace("universal_", "").replace("sparse_", "")
                print(f"\nUsing latest checkpoint: {checkpoint}")
                sweep_all(mode, epoch, universal=is_universal, sparse=is_sparse)
            else:
                print(f"No checkpoints found for {mode}")
        else:
            sweep_all(mode, args.epoch, universal=is_universal, sparse=is_sparse)