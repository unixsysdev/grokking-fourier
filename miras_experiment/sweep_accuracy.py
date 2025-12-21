import torch
from model_miras import UniversalFourierTransformer
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

def sweep_all(mode="ce", epoch="40000"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_check_dir = Path(f"miras_experiment/checkpoints/{mode}_sinpe")
    
    # Try both naming conventions
    path_options = [
        base_check_dir / f"{mode}_{epoch}.pt",
        base_check_dir / f"{mode}_e{epoch}.pt"
    ]
    
    model_path = None
    for p in path_options:
        if p.exists():
            model_path = p
            break
            
    if model_path is None:
        print(f"Error: Could not find checkpoint for {mode} {epoch} in {base_check_dir}")
        return
    
    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    primes = [p for p in range(2, 114) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    accuracies = {}
    for p in primes:
        acc = get_accuracy(model, p, device)
        accuracies[p] = acc
        
    print(f"--- Accuracy Sweep for {mode} {epoch} ---")
    sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    for p, acc in sorted_acc:
        if acc > 0.1 or p in [13, 71, 73, 79]:
            print(f"p={p:3d}: {acc:6.1%}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "ce"
    epoch = sys.argv[2] if len(sys.argv) > 2 else "40000"
    sweep_all(mode, epoch)
