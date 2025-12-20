"""
Deep analysis of specific Fourier-like dimensions in Qwen3.

This script focuses on dimensions that showed consistent Fourier structure
across layers (like dimension 35) and analyzes them in detail.
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


def create_prompt(a: int, b: int, p: int) -> str:
    """Create a modular addition prompt."""
    return f"{a} + {b} mod {p} ="


def extract_all_layer_activations(
    model,
    tokenizer,
    prompt: str,
) -> list:
    """Extract hidden states from ALL layers for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states[0] is embeddings, [1:] are layer outputs
    all_hidden = [h[0, -1, :].cpu().numpy() for h in outputs.hidden_states[1:]]
    return all_hidden


def collect_dimension_across_layers(
    model,
    tokenizer,
    p: int,
    dim_idx: int,
) -> np.ndarray:
    """
    Collect activations for a specific dimension across all layers.
    
    Returns:
        activations: (n_layers, p, p) array
    """
    n_layers = model.config.num_hidden_layers
    activations = np.zeros((n_layers, p, p))
    
    for a in tqdm(range(p), desc=f"Collecting dim {dim_idx}"):
        for b in range(p):
            prompt = create_prompt(a, b, p)
            all_hidden = extract_all_layer_activations(model, tokenizer, prompt)
            
            for layer_idx, hidden in enumerate(all_hidden):
                activations[layer_idx, a, b] = hidden[dim_idx]
    
    return activations


def analyze_dimension_detail(
    activations: np.ndarray,
    p: int,
    dim_idx: int,
    output_dir: Path,
):
    """
    Detailed analysis of a single dimension across all layers.
    
    Args:
        activations: (n_layers, p, p) array
        p: prime modulus
        dim_idx: which dimension this is
        output_dir: where to save results
    """
    n_layers = activations.shape[0]
    
    # Compute metrics for each layer
    layer_metrics = []
    
    for layer in range(n_layers):
        act_2d = activations[layer]
        
        # 2D FFT
        fft_2d = np.fft.fft2(act_2d)
        fft_mag = np.abs(fft_2d)
        
        # Sparsity (Gini)
        flat = fft_mag.flatten()
        flat_sorted = np.sort(flat)
        n = len(flat_sorted)
        gini = (2 * np.sum((np.arange(1, n+1) * flat_sorted))) / (n * np.sum(flat_sorted) + 1e-10) - (n + 1) / n
        
        # Reorganize by sum: for each s in [0, p-1], average where (a+b) mod p = s
        sum_activations = np.zeros(p)
        sum_counts = np.zeros(p)
        for a in range(p):
            for b in range(p):
                s = (a + b) % p
                sum_activations[s] += act_2d[a, b]
                sum_counts[s] += 1
        sum_activations /= sum_counts
        
        # 1D FFT of sum-organized activations
        fft_1d = np.fft.fft(sum_activations)
        fft_1d_mag = np.abs(fft_1d)
        
        # Dominant frequency (skip DC)
        dominant_freq = np.argmax(fft_1d_mag[1:p//2+1]) + 1
        
        # How well does cos(w_k * s) fit?
        best_r2 = 0
        best_freq = 0
        for k in range(1, p//2 + 1):
            cos_wave = np.cos(2 * np.pi * k * np.arange(p) / p)
            sin_wave = np.sin(2 * np.pi * k * np.arange(p) / p)
            
            # Fit: a*cos + b*sin + c
            X = np.column_stack([cos_wave, sin_wave, np.ones(p)])
            coeffs, _, _, _ = np.linalg.lstsq(X, sum_activations, rcond=None)
            
            predicted = X @ coeffs
            ss_res = np.sum((sum_activations - predicted) ** 2)
            ss_tot = np.sum((sum_activations - np.mean(sum_activations)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            if r2 > best_r2:
                best_r2 = r2
                best_freq = k
        
        layer_metrics.append({
            'layer': layer,
            'gini_2d': float(gini),
            'dominant_freq': int(dominant_freq),
            'best_fit_freq': int(best_freq),
            'best_fit_r2': float(best_r2),
            'sum_activations': sum_activations.tolist(),
            'fft_1d_mag': fft_1d_mag[:p//2+1].tolist(),
        })
    
    # Plot 1: Activation patterns across layers
    fig, axes = plt.subplots(4, 7, figsize=(24, 14))
    axes = axes.flatten()
    
    for layer in range(min(n_layers, 28)):
        ax = axes[layer]
        im = ax.imshow(activations[layer], cmap='RdBu_r', aspect='equal')
        ax.set_title(f'L{layer}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'Dimension {dim_idx}: Activation Patterns (a, b) Across All Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'dim{dim_idx}_activation_patterns.png', dpi=150)
    plt.close()
    
    # Plot 2: Sum-organized activations across layers
    fig, axes = plt.subplots(4, 7, figsize=(24, 14))
    axes = axes.flatten()
    
    for layer in range(min(n_layers, 28)):
        ax = axes[layer]
        sum_act = layer_metrics[layer]['sum_activations']
        ax.plot(sum_act, 'b-', linewidth=1)
        ax.set_title(f'L{layer} (R²={layer_metrics[layer]["best_fit_r2"]:.2f})', fontsize=9)
        ax.set_xlabel('(a+b) mod p', fontsize=8)
        if layer % 7 == 0:
            ax.set_ylabel('Activation', fontsize=8)
    
    plt.suptitle(f'Dimension {dim_idx}: Activation vs (a+b) mod {p}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'dim{dim_idx}_sum_activations.png', dpi=150)
    plt.close()
    
    # Plot 3: R² and Gini across layers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layers_arr = np.arange(n_layers)
    r2_vals = [m['best_fit_r2'] for m in layer_metrics]
    gini_vals = [m['gini_2d'] for m in layer_metrics]
    freq_vals = [m['best_fit_freq'] for m in layer_metrics]
    
    axes[0].plot(layers_arr, r2_vals, 'bo-')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('R² (best cosine fit)')
    axes[0].set_title(f'Dim {dim_idx}: Fourier Fit Quality')
    axes[0].axhline(0.5, color='r', linestyle='--', alpha=0.5, label='R²=0.5')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(layers_arr, gini_vals, 'go-')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Gini Coefficient')
    axes[1].set_title(f'Dim {dim_idx}: 2D FFT Sparsity')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(layers_arr, freq_vals, c='purple', s=50)
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Best-fit Frequency k')
    axes[2].set_title(f'Dim {dim_idx}: Dominant Frequency')
    axes[2].set_yticks(range(0, p//2 + 1, 2))
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'dim{dim_idx}_metrics_across_layers.png', dpi=150)
    plt.close()
    
    # Plot 4: FFT magnitudes across layers (heatmap)
    fft_matrix = np.array([m['fft_1d_mag'] for m in layer_metrics])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(fft_matrix.T, aspect='auto', cmap='hot')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Frequency k')
    ax.set_title(f'Dimension {dim_idx}: 1D FFT Magnitude Across Layers')
    plt.colorbar(im, ax=ax, label='FFT Magnitude')
    plt.tight_layout()
    plt.savefig(output_dir / f'dim{dim_idx}_fft_heatmap.png', dpi=150)
    plt.close()
    
    return layer_metrics


def find_fourier_dimensions(
    model,
    tokenizer,
    p: int,
    layer_idx: int,
    top_k: int = 20,
) -> list:
    """
    Find the most Fourier-like dimensions in a specific layer.
    Returns list of (dim_idx, sparsity_score) tuples.
    """
    hidden_size = model.config.hidden_size
    
    # Collect activations for all pairs (just for this layer)
    activations = np.zeros((p, p, hidden_size))
    
    for a in tqdm(range(p), desc=f"Scanning layer {layer_idx}"):
        for b in range(p):
            prompt = create_prompt(a, b, p)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
            activations[a, b, :] = hidden
    
    # For each dimension, compute Fourier fit quality
    dim_scores = []
    
    for d in range(hidden_size):
        act_2d = activations[:, :, d]
        
        # Organize by sum
        sum_activations = np.zeros(p)
        sum_counts = np.zeros(p)
        for a in range(p):
            for b in range(p):
                s = (a + b) % p
                sum_activations[s] += act_2d[a, b]
                sum_counts[s] += 1
        sum_activations /= sum_counts
        
        # 1D FFT
        fft_1d = np.fft.fft(sum_activations)
        fft_mag = np.abs(fft_1d)
        
        # Sparsity: ratio of max to mean (excluding DC)
        if fft_mag[1:].mean() > 1e-10:
            sparsity = fft_mag[1:].max() / fft_mag[1:].mean()
        else:
            sparsity = 0
        
        dim_scores.append((d, sparsity))
    
    # Sort by sparsity (higher = more Fourier-like)
    dim_scores.sort(key=lambda x: x[1], reverse=True)
    
    return dim_scores[:top_k]


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--p", type=int, default=23, help="Prime modulus")
    parser.add_argument("--dims", type=str, default="35,8,7,1,243",
                        help="Comma-separated dimension indices to analyze in detail")
    parser.add_argument("--output_dir", type=str, default="results_detailed")
    parser.add_argument("--find_dims", action="store_true",
                        help="Find top Fourier dimensions instead of analyzing specific ones")
    parser.add_argument("--find_layer", type=int, default=14,
                        help="Layer to use when finding Fourier dimensions")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model, tokenizer = load_qwen3(args.model)
    
    if args.find_dims:
        print(f"\nFinding top Fourier dimensions in layer {args.find_layer}...")
        top_dims = find_fourier_dimensions(model, tokenizer, args.p, args.find_layer)
        
        print("\nTop 20 most Fourier-like dimensions:")
        print(f"{'Dim':<10} {'Sparsity Score':<15}")
        print("-" * 25)
        for dim, score in top_dims:
            print(f"{dim:<10} {score:<15.4f}")
        
        # Save results
        with open(output_dir / 'top_fourier_dims.json', 'w') as f:
            json.dump({'layer': args.find_layer, 'top_dims': top_dims}, f, indent=2)
        
        # Use top 5 for detailed analysis
        dims_to_analyze = [d[0] for d in top_dims[:5]]
    else:
        dims_to_analyze = [int(d) for d in args.dims.split(",")]
    
    print(f"\nAnalyzing dimensions: {dims_to_analyze}")
    print(f"Prime modulus: {args.p}")
    
    all_dim_results = {}
    
    for dim_idx in dims_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing dimension {dim_idx}")
        print(f"{'='*60}")
        
        # Collect activations for this dimension across all layers
        activations = collect_dimension_across_layers(model, tokenizer, args.p, dim_idx)
        
        # Detailed analysis
        metrics = analyze_dimension_detail(activations, args.p, dim_idx, output_dir)
        all_dim_results[dim_idx] = metrics
        
        # Print summary for this dimension
        best_layer = max(metrics, key=lambda m: m['best_fit_r2'])
        print(f"Best R² = {best_layer['best_fit_r2']:.4f} at layer {best_layer['layer']}")
        print(f"Best frequency = {best_layer['best_fit_freq']}")
    
    # Save all results
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_dim_results)
    
    with open(output_dir / 'detailed_analysis.json', 'w') as f:
        json.dump({
            'model': args.model,
            'prime': args.p,
            'dimensions_analyzed': dims_to_analyze,
            'results': serializable_results
        }, f, indent=2)
    
    # Create comparison plot
    fig, axes = plt.subplots(len(dims_to_analyze), 1, figsize=(12, 3*len(dims_to_analyze)))
    if len(dims_to_analyze) == 1:
        axes = [axes]
    
    for idx, dim in enumerate(dims_to_analyze):
        ax = axes[idx]
        metrics = all_dim_results[dim]
        layers = [m['layer'] for m in metrics]
        r2_vals = [m['best_fit_r2'] for m in metrics]
        
        ax.plot(layers, r2_vals, 'o-', label=f'Dim {dim}')
        ax.set_ylabel('R² (Fourier fit)')
        ax.set_title(f'Dimension {dim}')
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    axes[-1].set_xlabel('Layer')
    plt.suptitle(f'Fourier Fit Quality Across Layers (p={args.p})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_dims_comparison.png', dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
