"""
Test whether Fourier structure is dynamic or hardcoded.

Questions:
1. Does dimension 35 generalize across different primes p?
2. Do different primes activate different Fourier dimensions?
3. Is there a "universal" modular arithmetic dimension, or separate ones per p?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def load_qwen3(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load Qwen3 model and tokenizer."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    
    return model, tokenizer


def extract_hidden_states(model, tokenizer, prompt: str, layer_idx: int = 14):
    """Extract hidden state from a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
    return hidden


def collect_activations_for_prime(model, tokenizer, p: int, layer_idx: int = 14):
    """Collect activations for all (a,b) pairs for a given prime."""
    hidden_size = model.config.hidden_size
    activations = np.zeros((p, p, hidden_size))
    
    for a in range(p):
        for b in range(p):
            prompt = f"{a} + {b} mod {p} ="
            hidden = extract_hidden_states(model, tokenizer, prompt, layer_idx)
            activations[a, b, :] = hidden
    
    return activations


def analyze_dimension_for_prime(activations, p, dim_idx):
    """
    Analyze how well a specific dimension fits cos(2πk(a+b)/p).
    Returns best R² and corresponding frequency.
    """
    act_2d = activations[:, :, dim_idx]
    
    # Organize by sum: average activation for each (a+b) mod p
    sum_activations = np.zeros(p)
    sum_counts = np.zeros(p)
    
    for a in range(p):
        for b in range(p):
            s = (a + b) % p
            sum_activations[s] += act_2d[a, b]
            sum_counts[s] += 1
    
    sum_activations /= sum_counts
    
    # Fit cosines at different frequencies
    best_r2 = 0
    best_freq = 0
    
    for k in range(1, p // 2 + 1):
        cos_wave = np.cos(2 * np.pi * k * np.arange(p) / p)
        sin_wave = np.sin(2 * np.pi * k * np.arange(p) / p)
        
        X = np.column_stack([cos_wave, sin_wave, np.ones(p)])
        coeffs, _, _, _ = np.linalg.lstsq(X, sum_activations, rcond=None)
        
        predicted = X @ coeffs
        ss_res = np.sum((sum_activations - predicted) ** 2)
        ss_tot = np.sum((sum_activations - np.mean(sum_activations)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
        
        if r2 > best_r2:
            best_r2 = r2
            best_freq = k
    
    return best_r2, best_freq, sum_activations


def find_top_fourier_dims(activations, p, top_k=10):
    """Find dimensions with strongest Fourier structure for a given prime."""
    hidden_size = activations.shape[2]
    
    dim_scores = []
    
    for d in range(hidden_size):
        r2, freq, _ = analyze_dimension_for_prime(activations, p, d)
        dim_scores.append({
            'dim': d,
            'r2': r2,
            'freq': freq
        })
    
    # Sort by R²
    dim_scores.sort(key=lambda x: x['r2'], reverse=True)
    
    return dim_scores[:top_k]


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--primes", type=str, default="7,11,13,17,19,23,29,31",
                        help="Comma-separated list of primes to test")
    parser.add_argument("--layer", type=int, default=14,
                        help="Layer to analyze")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model, tokenizer = load_qwen3(args.model)
    
    primes = [int(p) for p in args.primes.split(",")]
    
    print(f"\nTesting primes: {primes}")
    print(f"Layer: {args.layer}")
    print("="*60)
    
    # Track results
    all_results = {}
    dim35_results = []  # Track dim 35 specifically
    top_dims_per_prime = {}  # Track which dims are best for each prime
    
    for p in primes:
        print(f"\n{'='*60}")
        print(f"Testing p = {p}")
        print(f"{'='*60}")
        
        # Collect activations
        print(f"Collecting activations for {p}x{p} = {p*p} pairs...")
        activations = collect_activations_for_prime(model, tokenizer, p, args.layer)
        
        # Analyze dimension 35 specifically
        r2_35, freq_35, sum_act_35 = analyze_dimension_for_prime(activations, p, 35)
        dim35_results.append({
            'p': p,
            'r2': r2_35,
            'freq': freq_35,
            'sum_activations': sum_act_35.tolist()
        })
        print(f"\nDimension 35: R² = {r2_35:.4f}, freq = {freq_35}")
        
        # Find top Fourier dimensions for this prime
        print(f"\nFinding top Fourier dimensions...")
        top_dims = find_top_fourier_dims(activations, p, top_k=10)
        top_dims_per_prime[p] = top_dims
        
        print(f"\nTop 10 Fourier dimensions for p={p}:")
        print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
        print("-" * 26)
        for d in top_dims:
            marker = " <-- dim 35!" if d['dim'] == 35 else ""
            print(f"{d['dim']:<8} {d['r2']:<10.4f} {d['freq']:<8}{marker}")
        
        all_results[p] = {
            'dim35': {'r2': r2_35, 'freq': freq_35},
            'top_dims': top_dims
        }
    
    # Analysis: Does dim 35 generalize?
    print("\n" + "="*60)
    print("ANALYSIS: Does dimension 35 generalize across primes?")
    print("="*60)
    
    print(f"\n{'Prime':<8} {'Dim 35 R²':<12} {'Freq':<8} {'Rank of Dim 35':<15}")
    print("-" * 45)
    
    for p in primes:
        r2 = all_results[p]['dim35']['r2']
        freq = all_results[p]['dim35']['freq']
        
        # Find rank of dim 35
        rank = next((i+1 for i, d in enumerate(top_dims_per_prime[p]) if d['dim'] == 35), ">10")
        
        print(f"{p:<8} {r2:<12.4f} {freq:<8} {rank}")
    
    # Analysis: Which dimensions appear consistently?
    print("\n" + "="*60)
    print("ANALYSIS: Which dimensions appear in top 10 across primes?")
    print("="*60)
    
    dim_appearances = {}
    for p, top_dims in top_dims_per_prime.items():
        for d in top_dims:
            dim = d['dim']
            if dim not in dim_appearances:
                dim_appearances[dim] = []
            dim_appearances[dim].append({'p': p, 'r2': d['r2'], 'rank': top_dims.index(d) + 1})
    
    # Sort by number of appearances
    consistent_dims = sorted(dim_appearances.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n{'Dim':<8} {'Appearances':<12} {'Primes':<30} {'Avg R²':<10}")
    print("-" * 60)
    
    for dim, appearances in consistent_dims[:15]:
        n_appearances = len(appearances)
        primes_str = ",".join([str(a['p']) for a in appearances])
        avg_r2 = np.mean([a['r2'] for a in appearances])
        print(f"{dim:<8} {n_appearances:<12} {primes_str:<30} {avg_r2:<10.4f}")
    
    # Plot 1: Dim 35 across primes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, result in enumerate(dim35_results):
        if idx >= 8:
            break
        ax = axes[idx]
        p = result['p']
        sum_act = result['sum_activations']
        
        ax.plot(range(p), sum_act, 'bo-', markersize=4, label='Activation')
        
        # Add cosine fit
        k = result['freq']
        cos_wave = np.cos(2 * np.pi * k * np.arange(p) / p)
        sin_wave = np.sin(2 * np.pi * k * np.arange(p) / p)
        X = np.column_stack([cos_wave, sin_wave, np.ones(p)])
        coeffs, _, _, _ = np.linalg.lstsq(X, sum_act, rcond=None)
        fit = X @ coeffs
        ax.plot(range(p), fit, 'r-', alpha=0.7, label=f'Cosine fit (k={k})')
        
        ax.set_xlabel(f'(a+b) mod {p}')
        ax.set_ylabel('Activation')
        ax.set_title(f'p={p}, R²={result["r2"]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dimension 35: Does it generalize across primes?', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'dim35_across_primes.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'dim35_across_primes.png'}")
    
    # Plot 2: Which dimensions are best for each prime?
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap of top dimensions per prime
    all_top_dims = set()
    for p, top_dims in top_dims_per_prime.items():
        for d in top_dims[:5]:  # Top 5
            all_top_dims.add(d['dim'])
    
    all_top_dims = sorted(all_top_dims)
    
    heatmap = np.zeros((len(primes), len(all_top_dims)))
    
    for i, p in enumerate(primes):
        top_dims = top_dims_per_prime[p]
        for d in top_dims:
            if d['dim'] in all_top_dims:
                j = all_top_dims.index(d['dim'])
                heatmap[i, j] = d['r2']
    
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(all_top_dims)))
    ax.set_xticklabels(all_top_dims, rotation=45)
    ax.set_yticks(range(len(primes)))
    ax.set_yticklabels(primes)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Prime p')
    ax.set_title('Top Fourier Dimensions by Prime (R² values)')
    plt.colorbar(im, ax=ax, label='R²')
    
    # Highlight dim 35
    if 35 in all_top_dims:
        j = all_top_dims.index(35)
        ax.axvline(j - 0.5, color='blue', linewidth=2, linestyle='--')
        ax.axvline(j + 0.5, color='blue', linewidth=2, linestyle='--')
        ax.text(j, -0.7, 'dim 35', ha='center', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_dims_per_prime.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'top_dims_per_prime.png'}")
    
    # Plot 3: R² of dim 35 vs prime
    fig, ax = plt.subplots(figsize=(10, 6))
    
    primes_arr = [r['p'] for r in dim35_results]
    r2_arr = [r['r2'] for r in dim35_results]
    
    ax.bar(range(len(primes_arr)), r2_arr, color='steelblue')
    ax.set_xticks(range(len(primes_arr)))
    ax.set_xticklabels(primes_arr)
    ax.set_xlabel('Prime p')
    ax.set_ylabel('R² for Dimension 35')
    ax.set_title('Does Dimension 35 Generalize? R² by Prime')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.9')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.legend()
    ax.set_ylim(0, 1)
    
    for i, r2 in enumerate(r2_arr):
        ax.text(i, r2 + 0.02, f'{r2:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dim35_r2_by_prime.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'dim35_r2_by_prime.png'}")
    
    # Save all results
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'primes': primes,
            'layer': args.layer,
            'dim35_results': convert(dim35_results),
            'top_dims_per_prime': convert(top_dims_per_prime),
            'consistent_dims': convert(consistent_dims[:20])
        }, f, indent=2)
    
    print(f"\nSaved: {output_dir / 'results.json'}")
    
    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    avg_r2_dim35 = np.mean(r2_arr)
    std_r2_dim35 = np.std(r2_arr)
    
    print(f"\nDimension 35 across all primes:")
    print(f"  Average R²: {avg_r2_dim35:.4f}")
    print(f"  Std dev:    {std_r2_dim35:.4f}")
    print(f"  Min R²:     {min(r2_arr):.4f} (p={primes_arr[np.argmin(r2_arr)]})")
    print(f"  Max R²:     {max(r2_arr):.4f} (p={primes_arr[np.argmax(r2_arr)]})")
    
    if avg_r2_dim35 > 0.8 and std_r2_dim35 < 0.1:
        print("\n→ DYNAMIC: Dim 35 generalizes well across primes!")
    elif avg_r2_dim35 > 0.5:
        print("\n→ PARTIALLY DYNAMIC: Dim 35 shows some generalization but inconsistent")
    else:
        print("\n→ HARDCODED: Dim 35 does not generalize; may be tuned to specific p")
    
    # Check if different dims dominate for different primes
    best_dims = [top_dims_per_prime[p][0]['dim'] for p in primes]
    unique_best = set(best_dims)
    
    print(f"\nBest dimension by prime: {dict(zip(primes, best_dims))}")
    print(f"Number of unique 'best' dimensions: {len(unique_best)}")
    
    if len(unique_best) == 1:
        print(f"\n→ UNIVERSAL: Same dimension ({best_dims[0]}) is best for all primes!")
    elif len(unique_best) <= len(primes) // 2:
        print(f"\n→ SEMI-UNIVERSAL: A few dimensions handle most primes")
    else:
        print(f"\n→ SPECIALIZED: Different primes activate different dimensions")


if __name__ == "__main__":
    main()
