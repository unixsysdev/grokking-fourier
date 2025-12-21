import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import math
from einops import einsum, rearrange
from model_miras import UniversalFourierTransformer
from tqdm import tqdm

def dft_2d(activations):
    """Compute 2D Discrete Fourier Transform."""
    p_size = activations.shape[0]
    # We pad to a larger size to get smoother plots
    n_fft = 64 
    fft_res = np.fft.fft2(activations, s=(n_fft, n_fft))
    return np.abs(np.fft.fftshift(fft_res))

def get_activations(model, p, device):
    """Collect the MLP hidden activations for all (a,b) pairs for a given p."""
    model.eval()
    
    # We catch the MLP hidden state of Layer 2
    mlp_acts = []
    def hook_fn(module, input, output):
        # module is model.layer2.mlp[0] (the first Linear layer)
        h = torch.relu(output)
        mlp_acts.append(h.detach().cpu())

    # In model_miras.py: self.layer2.mlp is nn.Sequential(Linear, ReLU, Linear)
    handle = model.layer2.mlp[0].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        a_vals = torch.arange(p, device=device).repeat_interleave(p)
        b_vals = torch.arange(p, device=device).repeat(p)
        p_vals = torch.full_like(a_vals, p)
        
        # Process in batches
        batch_size = 512
        for i in range(0, len(a_vals), batch_size):
            end = min(i + batch_size, len(a_vals))
            model(p_vals[i:end], a_vals[i:end], b_vals[i:end])
            
    handle.remove()
    
    # mlp_acts is a list of [batch, seq=4, d_mlp]
    # We care about the final prediction position (index 3)
    all_acts = torch.cat(mlp_acts, dim=0)[:, 3, :] # (p*p, d_mlp)
    return all_acts.view(p, p, -1)

def plot_fourier_analysis(model, p, device, output_dir):
    print(f"Analyzing Fourier patterns for mod {p}...")
    acts = get_activations(model, p, device) # (p, p, d_mlp)
    d_mlp = acts.shape[-1]
    
    n_plot = 3
    # Find top neurons for this prime (by variance or simple max act)
    variances = acts.var(dim=(0, 1))
    top_neurons = variances.argsort(descending=True)[:n_plot]
    
    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 4*n_plot))
    for i, neuron_idx in enumerate(top_neurons):
        neuron_acts = acts[:, :, neuron_idx].numpy()
        
        # 1. Spatial Plot
        sns.heatmap(neuron_acts, ax=axes[i, 0], cmap="RdBu_r", center=0)
        axes[i, 0].set_title(f"Neuron {neuron_idx} Activations (mod {p})")
        
        # 2. Fourier Plot
        fourier = dft_2d(neuron_acts)
        axes[i, 1].imshow(fourier, cmap="viridis")
        axes[i, 1].set_title(f"2D Fourier Spectrum (mod {p})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"fourier_mod_{p}.png")
    plt.close()
    return top_neurons

def causal_ablation(model, p, target_neurons, device):
    """Zero out neurons and measure accuracy drop for a specific p."""
    model.eval()
    
    def get_acc(ablated_indices=None):
        correct = 0
        total = p * p
        
        with torch.no_grad():
            a = torch.arange(p, device=device).repeat_interleave(p)
            b = torch.arange(p, device=device).repeat(p)
            p_v = torch.full_like(a, p)
            
            # Manual forward matching UniversalFourierTransformer
            eq_token = model.max_p + 1
            tokens = torch.stack([p_v, a, b, torch.full_like(a, eq_token)], dim=1)
            x = model.token_embed(tokens) + model.pos_embed(torch.arange(4, device=device))
            
            # 1. MIRAS Update
            z_p = model.memory_block(x[:, 0, :])
            x[:, 0, :] = model.mem_to_hidden(z_p)
            
            # 2. Transformer layers
            for layer in [model.layer1, model.layer2]:
                # Attention
                norm_x = layer.ln1(x)
                q = einsum(norm_x, layer.W_Q, 'b s d, h d k -> b h s k')
                k = einsum(norm_x, layer.W_K, 'b s d, h d k -> b h s k')
                v = einsum(norm_x, layer.W_V, 'b s d, h d k -> b h s k')
                
                scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(layer.d_head)
                attn = torch.softmax(scores, dim=-1)
                out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
                out = einsum(out, layer.W_O, 'b h s d, h d m -> b s m')
                x = x + out
                
                # MLP with potential ablation
                norm_x2 = layer.ln2(x)
                # layer.mlp is Sequential(Linear, ReLU, Linear)
                hidden = torch.relu(layer.mlp[0](norm_x2))
                
                # Only ablate in layer 2 for consistency with hook
                if ablated_indices is not None and layer == model.layer2:
                    hidden[:, :, ablated_indices] = 0
                
                mlp_out = layer.mlp[2](hidden)
                x = x + mlp_out
            
            # 3. Pred from index 3
            logits = model.unembed(x[:, 3, :])
            preds = logits.argmax(dim=-1)
            correct = (preds == (a + b) % p).float().sum().item()
            return correct / total

    baseline = get_acc()
    ablated = get_acc(target_neurons)
    return baseline, ablated

def analyze_attention_flow(model, p, device):
    """Measure how much '=' attends to 'p'."""
    model.eval()
    tokens = torch.tensor([[p, 0, 0, model.max_p + 1]], device=device) # (a,b) arbitrary for p-flow
    
    flow_stats = []
    with torch.no_grad():
        x = model.token_embed(tokens) + model.pos_embed(torch.arange(4, device=device))
        z_p = model.memory_block(x[:, 0, :])
        x[:, 0, :] = model.mem_to_hidden(z_p)
        
        for i, layer in enumerate([model.layer1, model.layer2]):
            norm_x = layer.ln1(x)
            q = einsum(norm_x, layer.W_Q, 'b s d, h d k -> b h s k')
            k = einsum(norm_x, layer.W_K, 'b s d, h d k -> b h s k')
            scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(layer.d_head)
            attn = torch.softmax(scores, dim=-1)
            
            # Attn from '=' (index 3) to all positions [p, a, b, =]
            eq_attn = attn[0, :, 3, :].cpu().numpy() # (n_heads, 4)
            flow_stats.append(eq_attn)
            
            # Forward for next layer
            v = einsum(norm_x, layer.W_V, 'b s d, h d k -> b h s k')
            out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
            out = einsum(out, layer.W_O, 'b h s d, h d m -> b s m')
            x = x + out
            x = x + layer.mlp(layer.ln2(x))
            
    return np.array(flow_stats) # (layers, heads, 4)

def calculate_fourier_snr_1d(activations, p):
    """
    Measure the 'purity' of the Fourier signal in 1D.
    We correlate activations with (a+b)%p.
    """
    # activations: (p*p, 1) or (p, p)
    # We flatten and sort by (a+b)%p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    sums = (a + b) % p
    
    # Average activation for each possible sum value [0, ..., p-1]
    avg_acts = np.zeros(p)
    acts_flat = activations.reshape(-1)
    for s in range(p):
        mask = (sums == s)
        if mask.any():
            avg_acts[s] = acts_flat[mask].mean()
            
    # Compute 1D FFT
    fft_vals = np.abs(np.fft.fft(avg_acts))
    # Fourier SNR: Max non-zero freq component / sum of others
    # (Ignore DC component at index 0)
    if len(fft_vals) > 1:
        energy = fft_vals[1:]
        max_idx = np.argmax(energy)
        max_val = energy[max_idx]
        noise = energy.sum() - max_val
        snr = max_val / (noise + 1e-9)
        return snr, max_idx + 1 # +1 because we sliced 1:
    return 0, 0

def plot_activation_vs_sum(model, p, device, output_dir):
    """Nanda-style 1D plot: Activation vs (a+b)%p."""
    acts = get_activations(model, p, device) # (p, p, d_mlp)
    d_mlp = acts.shape[-1]
    
    # Pick top neuron
    variances = acts.var(dim=(0, 1))
    top_idx = variances.argmax().item()
    neuron_acts = acts[:, :, top_idx].numpy()
    
    # Prep data for 1D plot
    a = np.arange(p).repeat(p)
    b = np.tile(np.arange(p), p)
    sums = (a + b) % p
    flat_acts = neuron_acts.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sums, flat_acts, alpha=0.3, s=10)
    
    # Line of means
    means = [flat_acts[sums == s].mean() for s in range(p)]
    plt.plot(range(p), means, color='red', linewidth=2, label='Mean Activation')
    
    plt.title(f"Neuron {top_idx} Activation vs (a+b) mod {p}")
    plt.xlabel("(a + b) mod p")
    plt.ylabel("Activation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"nanda_signal_mod_{p}.png")
    plt.close()
    
    snr, freq = calculate_fourier_snr_1d(neuron_acts, p)
    return snr, freq

def extrapolation_stress_test(model, device, output_dir):
    """Test performance on larger and larger primes (extrapolation)."""
    print("\n--- Starting Extrapolation Stress Test ---")
    
    # Range of primes to test (beyond the training set of ~23 primes up to 79)
    # Architectural limit is max_p=150
    test_primes = [p for p in range(83, 151) if all(p % i != 0 for i in range(2, int(p**0.5) + 1))]
    
    results = {"p": [], "accuracy": [], "purity_snr": []}
    
    for p in tqdm(test_primes):
        # 1. Accuracy
        acc, _ = causal_ablation(model, p, None, device) # Passing None returns baseline
        
        # 2. Fourier Purity (on a sample of neurons or top neuron)
        acts = get_activations(model, p, device)
        variances = acts.var(dim=(0, 1))
        top_idx = variances.argmax().item()
        snr, _ = calculate_fourier_snr_1d(acts[:, :, top_idx].numpy(), p)
        
        results["p"].append(p)
        results["accuracy"].append(acc)
        results["purity_snr"].append(snr)

    # Plotting results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Modulus (p)')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(results["p"], results["accuracy"], 'o-', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Fourier SNR', color=color)
    ax2.plot(results["p"], results["purity_snr"], 's--', color=color, label='Purity SNR', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Universal Engine Extrapolation Stress Test')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "extrapolation_decay_curve.png")
    plt.close()

    with open(output_dir / "extrapolation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Stress test complete. Curve saved to {output_dir}/extrapolation_decay_curve.png")
    
    # Find the breakpoint (e.g., accuracy < 50%)
    for i, acc in enumerate(results["accuracy"]):
        if acc < 0.5:
            print(f"CRITICAL: Performance breakpoint detected at p={results['p'][i]} (Acc: {acc:.1%})")
            break

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ce", choices=["ce", "rl"], help="Analyze CE or RL model")
    parser.add_argument("--epoch", type=int, default=-1, help="Specific epoch to load (-1 for latest)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    base_check_dir = Path(f"miras_experiment/checkpoints/{args.mode}_sinpe")
    prefix = f"{args.mode}_e"
    
    checkpoints = sorted(list(base_check_dir.glob(f"{prefix}*.pt")), 
                         key=lambda x: int(x.stem.split('_e')[1]) if len(x.stem.split('_e')) > 1 else 0)
    
    if not checkpoints:
        print(f"No checkpoints found in {base_check_dir}")
        return
        
    if args.epoch == -1:
        model_path = checkpoints[-1]
    else:
        model_path = base_check_dir / f"{prefix}{args.epoch}.pt"
        if not model_path.exists():
            print(f"Checkpoint {model_path} does not exist.")
            return

    history_path = base_check_dir / "history.json"
    
    print(f"Loading checkpoint for analysis: {model_path}")
    
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        print("Warning: history.json not found.")
        history = {}
    
    # Primes from the training script
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    # In SIN-PE, we want to test even larger ones to see if the cliff moved
    seen_p = 13
    unseen_p = 71
    extrapolation_p = 101 # Far beyond training range [2, 79]
    
    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    output_dir = Path("miras_experiment/analysis")
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for p_label, p in [("seen", seen_p), ("unseen", unseen_p), ("extrap", extrapolation_p)]:
        print(f"\n--- Analyzing Prime {p} ({p_label}) ---")
        results[f"{p_label}_{p}"] = {}
        
        # 1. Fourier Analysis (2D)
        top_neurons = plot_fourier_analysis(model, p, device, output_dir)
        
        # 2. Nanda-style 1D Analysis
        snr, freq = plot_activation_vs_sum(model, p, device, output_dir)
        print(f"Fourier Purity (SNR): {snr:.2f} (Freq k={freq})")
        results[f"{p_label}_{p}"]["purity_snr"] = float(snr)
        
        # 3. Causality
        base, abl = causal_ablation(model, p, top_neurons, device)
        results[f"{p_label}_{p}"] = {**results[f"{p_label}_{p}"], "baseline": base, "ablated": abl, "drop": base - abl}
        print(f"Causal Ablation: Baseline {base:.1%}, Ablated {abl:.1%}, Drop {base-abl:.1%}")
        
        # 4. Attention Flow
        flow = analyze_attention_flow(model, p, device) # (layers, heads, 4)
        l1_p_attn = flow[0, :, 0].mean()
        l2_p_attn = flow[1, :, 0].mean()
        print(f"Mean '=' to 'p' Attention: Layer 1: {l1_p_attn:.2f}, Layer 2: {l2_p_attn:.2f}")
        results[f"{p_label}_{p}"]["p_attn"] = [float(l1_p_attn), float(l2_p_attn)]

    with open(output_dir / f"latest_{args.mode}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    
    # 4. Run Stress Test
    extrapolation_stress_test(model, device, output_dir)

if __name__ == "__main__":
    main()
