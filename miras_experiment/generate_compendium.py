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
    # Handle universal mode
    if mode == "universal":
        base_check_dir = Path(f"checkpoints/ce_universal")
        path_options = [
            base_check_dir / f"universal_{epoch}.pt",
            base_check_dir / f"universal_e{epoch}.pt"
        ]
        max_p, d_model, n_heads, d_mlp, d_mem = 200, 256, 8, 1024, 256
    else:
        base_check_dir = Path(f"checkpoints/{mode}_sinpe")
        path_options = [
            base_check_dir / f"{mode}_{epoch}.pt",
            base_check_dir / f"{mode}_e{epoch}.pt"
        ]
        max_p, d_model, n_heads, d_mlp, d_mem = 150, 128, 4, 512, 128
    
    checkpoint_path = None
    for p in path_options:
        if p.exists():
            checkpoint_path = p
            break

    if checkpoint_path is None:
        print(f"Error: Could not find checkpoint for {mode} {epoch} in {base_check_dir}")
        return

    output_dir = Path(f"analysis/compendium/{mode}_{epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Generating Compendium for {mode} {epoch} ---")
    
    # 1. Load Model
    model = UniversalFourierTransformer(max_p=max_p, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp, d_mem=d_mem).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Fourier & Nanda Plots (Seen, Unseen, Extrapolation)
    if mode == "universal":
        # Test interpolation and extrapolation for universal model
        test_moduli = [13, 45, 95, 121, 149]  # mix of train range, held-out, and extrapolation
    else:
        test_moduli = [13, 71, 101]
        
    for p in test_moduli:
        print(f"Analyzing m={p}...")
        
        # 2D Fourier (takes model, p, device, output_dir)
        analyze.plot_fourier_analysis(model, p, device, output_dir)
        
        # 1D Nanda Signal (takes model, p, device, output_dir)
        analyze.plot_activation_vs_sum(model, p, device, output_dir)
        
        # Attention Flow
        flow = analyze.analyze_attention_flow(model, p, device)
        avg_l1 = flow[0, :, 0].mean()
        avg_l2 = flow[1, :, 0].mean()
        with open(output_dir / f"attn_flow_m{p}.txt", "w") as f:
            f.write(f"Layer 1 Modulus Attn: {avg_l1:.4f}\n")
            f.write(f"Layer 2 Modulus Attn: {avg_l2:.4f}\n")

    # 3. Neuron Adaptation (Cross-Prime)
    print("Generating Neuron Adaptation plots...")
    if mode == "universal":
        adapt_moduli = [11, 25, 50, 100]  # mix for universal
    else:
        adapt_moduli = [11, 17, 23, 31]
    prime_top = adapt.find_top_fourier_overlap(model, adapt_moduli, device)
    
    all_sets = list(prime_top.values())
    intersection = list(set.intersection(*all_sets))
    to_plot = intersection[:3] if len(intersection) >= 3 else list(prime_top[adapt_moduli[0]])[:3]
    
    for n_idx in tqdm.tqdm(to_plot):
        adapt.plot_neuron_cross_prime(model, adapt_moduli, n_idx, device, output_dir)

    # 4. Extrapolation Decay Curve
    print("Generating Extrapolation Decay Curve...")
    analyze.extrapolation_stress_test(model, device, output_dir)

    print(f"Compendium complete: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ce", "rl", "universal"])
    parser.add_argument("epoch", type=str)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    generate_compendium(args.mode, args.epoch, device)