import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from model_miras import UniversalFourierTransformer
import analyze_miras_mechanics as analyze
import visualize_adaptation as adapt
import tqdm

def generate_compendium(mode, epoch, device):
    base_check_dir = Path(f"miras_experiment/checkpoints/{mode}_sinpe")
    path_options = [
        base_check_dir / f"{mode}_{epoch}.pt",
        base_check_dir / f"{mode}_e{epoch}.pt"
    ]
    
    checkpoint_path = None
    for p in path_options:
        if p.exists():
            checkpoint_path = p
            break

    if checkpoint_path is None:
        print(f"Error: Could not find checkpoint for {mode} {epoch} in {base_check_dir}")
        return

    output_dir = Path(f"miras_experiment/analysis/compendium/{mode}_{epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Generating Compendium for {mode} {epoch} ---")
    
    # 1. Load Model
    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Fourier & Nanda Plots (Seen, Unseen, Extrapolation)
    test_primes = [13, 71, 101]
    for p in test_primes:
        print(f"Analyzing p={p}...")
        
        # 2D Fourier (takes model, p, device, output_dir)
        analyze.plot_fourier_analysis(model, p, device, output_dir)
        
        # 1D Nanda Signal (takes model, p, device, output_dir)
        analyze.plot_activation_vs_sum(model, p, device, output_dir)
        
        # Attention Flow
        flow = analyze.analyze_attention_flow(model, p, device)
        avg_l1 = flow[0, :, 0].mean()
        avg_l2 = flow[1, :, 0].mean()
        with open(output_dir / f"attn_flow_p{p}.txt", "w") as f:
            f.write(f"Layer 1 Modulus Attn: {avg_l1:.4f}\n")
            f.write(f"Layer 2 Modulus Attn: {avg_l2:.4f}\n")

    # 3. Neuron Adaptation (Cross-Prime)
    print("Generating Neuron Adaptation plots...")
    adapt_primes = [11, 17, 23, 31]
    prime_top = adapt.find_top_fourier_overlap(model, adapt_primes, device)
    
    all_sets = list(prime_top.values())
    intersection = list(set.intersection(*all_sets))
    to_plot = intersection[:3] if len(intersection) >= 3 else list(prime_top[adapt_primes[0]])[:3]
    
    for n_idx in tqdm.tqdm(to_plot):
        adapt.plot_neuron_cross_prime(model, adapt_primes, n_idx, device, output_dir)

    # 4. Extrapolation Decay Curve
    print("Generating Extrapolation Decay Curve...")
    analyze.extrapolation_stress_test(model, device, output_dir)

    print(f"Compendium complete: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ce", "rl"])
    parser.add_argument("epoch", type=str)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generate_compendium(args.mode, args.epoch, device)
