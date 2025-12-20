"""
Qwen3 0.6B Fourier Analysis for Modular Arithmetic

Investigates whether a pretrained LLM shows similar Fourier structure
to the small transformers in the grokking paper when doing modular arithmetic.

Experiment:
1. Run Qwen3 on "a + b mod p = " for all pairs (a, b)
2. Extract MLP activations from various layers
3. Apply 2D Fourier transform
4. Look for sparse frequency structure
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
        torch_dtype=torch.float32,  # Use float32 for analysis precision
        device_map="auto",
    )
    model.eval()
    
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    
    return model, tokenizer


def create_prompt(a: int, b: int, p: int) -> str:
    """Create a modular addition prompt."""
    return f"{a} + {b} mod {p} ="


def extract_mlp_activations(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract MLP activations from a specific layer for a given prompt.
    
    Returns the MLP output (after the activation function) for the last token.
    """
    activations = {}
    
    def hook_fn(module, input, output):
        # output shape: (batch, seq_len, intermediate_size) for gate/up proj
        # or (batch, seq_len, hidden_size) for down proj
        activations['mlp_output'] = output.detach()
    
    # Register hook on the MLP's down_proj (final output of MLP block)
    # In Qwen3, the MLP structure is: gate_proj, up_proj -> activation -> down_proj
    handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook_fn)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        
        # Get activation for the last token (where we'd predict the answer)
        mlp_out = activations['mlp_output'][0, -1, :]  # (hidden_size,)
        return mlp_out.cpu()
    finally:
        handle.remove()


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer for a given prompt.
    
    Returns the hidden state for the last token.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states is a tuple of (num_layers + 1,) tensors
    # Each tensor is (batch, seq_len, hidden_size)
    hidden = outputs.hidden_states[layer_idx + 1][0, -1, :]  # +1 because index 0 is embeddings
    return hidden.cpu()


def collect_activations_grid(
    model,
    tokenizer,
    p: int,
    layer_idx: int,
    activation_type: str = "mlp",  # "mlp" or "hidden"
    max_pairs: int = None,
) -> np.ndarray:
    """
    Collect activations for all (a, b) pairs.
    
    Returns:
        activations: (p, p, hidden_size) array
    """
    if activation_type == "mlp":
        extract_fn = lambda prompt: extract_mlp_activations(model, tokenizer, prompt, layer_idx)
        size = model.config.hidden_size
    else:
        extract_fn = lambda prompt: extract_hidden_states(model, tokenizer, prompt, layer_idx)
        size = model.config.hidden_size
    
    activations = np.zeros((p, p, size))
    
    pairs = [(a, b) for a in range(p) for b in range(p)]
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    for a, b in tqdm(pairs, desc=f"Layer {layer_idx} {activation_type}"):
        prompt = create_prompt(a, b, p)
        act = extract_fn(prompt)
        activations[a, b, :] = act.numpy()
    
    return activations


def fourier_analysis_2d(activations: np.ndarray, p: int) -> dict:
    """
    Perform 2D Fourier analysis on the activation grid.
    
    Args:
        activations: (p, p, hidden_size) array
        p: prime modulus
    
    Returns:
        Dictionary with Fourier analysis results
    """
    hidden_size = activations.shape[-1]
    
    # Compute 2D FFT for each hidden dimension
    fft_magnitudes = np.zeros((p, p))
    
    for d in range(hidden_size):
        act_2d = activations[:, :, d]
        fft_2d = np.fft.fft2(act_2d)
        fft_magnitudes += np.abs(fft_2d) ** 2
    
    # Average over dimensions
    fft_magnitudes = np.sqrt(fft_magnitudes / hidden_size)
    
    # Find significant frequencies (above threshold)
    threshold = np.mean(fft_magnitudes) + 2 * np.std(fft_magnitudes)
    significant = fft_magnitudes > threshold
    
    # Compute sparsity (Gini coefficient)
    flat = fft_magnitudes.flatten()
    flat_sorted = np.sort(flat)
    n = len(flat_sorted)
    cumsum = np.cumsum(flat_sorted)
    gini = (2 * np.sum((np.arange(1, n+1) * flat_sorted))) / (n * np.sum(flat_sorted)) - (n + 1) / n
    
    return {
        'fft_magnitudes': fft_magnitudes,
        'significant_mask': significant,
        'n_significant': np.sum(significant),
        'gini_coefficient': gini,
        'threshold': threshold,
    }


def fourier_analysis_1d_per_dim(activations: np.ndarray, p: int, top_k: int = 20) -> dict:
    """
    Analyze each hidden dimension separately with 1D Fourier over (a+b) mod p.
    
    This is closer to what the grokking paper does - looking for dimensions
    that represent cos(w_k * (a+b)) or sin(w_k * (a+b)).
    """
    hidden_size = activations.shape[-1]
    
    # For each (a, b) pair, compute (a + b) mod p
    # Then for each hidden dim, see if it correlates with specific frequencies
    
    # Reorganize by sum: for each s in [0, p-1], average activations where (a+b) mod p = s
    sum_activations = np.zeros((p, hidden_size))
    sum_counts = np.zeros(p)
    
    for a in range(p):
        for b in range(p):
            s = (a + b) % p
            sum_activations[s, :] += activations[a, b, :]
            sum_counts[s] += 1
    
    sum_activations /= sum_counts[:, np.newaxis]
    
    # Now do 1D FFT for each hidden dimension
    fft_per_dim = np.zeros((hidden_size, p))
    for d in range(hidden_size):
        fft_1d = np.fft.fft(sum_activations[:, d])
        fft_per_dim[d, :] = np.abs(fft_1d)
    
    # Find dimensions that are "Fourier-like" (sparse in frequency domain)
    # Compute sparsity per dimension
    sparsity_per_dim = []
    for d in range(hidden_size):
        magnitudes = fft_per_dim[d, :]
        # Normalized entropy as sparsity measure (lower = sparser)
        magnitudes_norm = magnitudes / (magnitudes.sum() + 1e-10)
        entropy = -np.sum(magnitudes_norm * np.log(magnitudes_norm + 1e-10))
        max_entropy = np.log(p)
        sparsity_per_dim.append(1 - entropy / max_entropy)
    
    sparsity_per_dim = np.array(sparsity_per_dim)
    
    # Find top-k most Fourier-like dimensions
    top_dims = np.argsort(sparsity_per_dim)[-top_k:][::-1]
    
    # For each top dimension, find dominant frequency
    dominant_freqs = []
    for d in top_dims:
        freq = np.argmax(fft_per_dim[d, 1:]) + 1  # Skip DC component
        dominant_freqs.append(freq)
    
    return {
        'sum_activations': sum_activations,
        'fft_per_dim': fft_per_dim,
        'sparsity_per_dim': sparsity_per_dim,
        'top_fourier_dims': top_dims,
        'dominant_freqs': dominant_freqs,
        'top_sparsities': sparsity_per_dim[top_dims],
    }


def plot_results(
    results_2d: dict,
    results_1d: dict,
    layer_idx: int,
    p: int,
    output_dir: Path,
    activation_type: str,
):
    """Generate and save analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: 2D FFT magnitude
    ax = axes[0, 0]
    im = ax.imshow(np.log10(results_2d['fft_magnitudes'] + 1e-10), cmap='hot')
    ax.set_title(f'2D FFT Magnitude (log10)\nLayer {layer_idx}, {activation_type}')
    ax.set_xlabel('Frequency k_b')
    ax.set_ylabel('Frequency k_a')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Significant frequencies
    ax = axes[0, 1]
    ax.imshow(results_2d['significant_mask'], cmap='Blues')
    ax.set_title(f"Significant Frequencies: {results_2d['n_significant']}\nGini: {results_2d['gini_coefficient']:.3f}")
    ax.set_xlabel('Frequency k_b')
    ax.set_ylabel('Frequency k_a')
    
    # Plot 3: Sparsity distribution across dimensions
    ax = axes[1, 0]
    ax.hist(results_1d['sparsity_per_dim'], bins=50, edgecolor='black')
    ax.axvline(results_1d['top_sparsities'][-1], color='red', linestyle='--', 
               label=f'Top-{len(results_1d["top_fourier_dims"])} threshold')
    ax.set_xlabel('Fourier Sparsity (higher = more sparse)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fourier Sparsity Across Dimensions')
    ax.legend()
    
    # Plot 4: Top Fourier-like dimensions
    ax = axes[1, 1]
    top_k = min(10, len(results_1d['top_fourier_dims']))
    for i in range(top_k):
        dim = results_1d['top_fourier_dims'][i]
        fft_mag = results_1d['fft_per_dim'][dim, :p//2]
        ax.plot(fft_mag, label=f'Dim {dim} (freq={results_1d["dominant_freqs"][i]})', alpha=0.7)
    
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('FFT Magnitude')
    ax.set_title(f'Top {top_k} Most Fourier-like Dimensions')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'fourier_analysis_layer{layer_idx}_{activation_type}.png', dpi=150)
    plt.close()
    
    # Additional plot: Activation patterns for top dimensions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, dim in enumerate(results_1d['top_fourier_dims'][:10]):
        ax = axes[i]
        # We need the original activations for this - pass them in
        ax.plot(results_1d['sum_activations'][:, dim])
        ax.set_title(f'Dim {dim}\nFreq {results_1d["dominant_freqs"][i]}, Sparsity {results_1d["top_sparsities"][i]:.3f}')
        ax.set_xlabel('(a+b) mod p')
        ax.set_ylabel('Activation')
    
    plt.suptitle(f'Activation vs (a+b) mod {p} for Top Fourier Dimensions\nLayer {layer_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'top_dims_layer{layer_idx}_{activation_type}.png', dpi=150)
    plt.close()


def analyze_layer(
    model,
    tokenizer,
    p: int,
    layer_idx: int,
    output_dir: Path,
    activation_type: str = "mlp",
):
    """Run full analysis for a single layer."""
    print(f"\n{'='*60}")
    print(f"Analyzing Layer {layer_idx} ({activation_type})")
    print(f"{'='*60}")
    
    # Collect activations
    activations = collect_activations_grid(
        model, tokenizer, p, layer_idx, activation_type
    )
    
    # 2D Fourier analysis
    results_2d = fourier_analysis_2d(activations, p)
    print(f"2D FFT - Significant frequencies: {results_2d['n_significant']}")
    print(f"2D FFT - Gini coefficient: {results_2d['gini_coefficient']:.4f}")
    
    # 1D Fourier analysis per dimension
    results_1d = fourier_analysis_1d_per_dim(activations, p)
    print(f"Top Fourier dimensions: {results_1d['top_fourier_dims'][:5]}")
    print(f"Their dominant frequencies: {results_1d['dominant_freqs'][:5]}")
    print(f"Their sparsities: {results_1d['top_sparsities'][:5]}")
    
    # Plot results
    plot_results(results_2d, results_1d, layer_idx, p, output_dir, activation_type)
    
    return {
        'layer': layer_idx,
        'activation_type': activation_type,
        'n_significant_2d': int(results_2d['n_significant']),
        'gini_2d': float(results_2d['gini_coefficient']),
        'top_dims': [int(x) for x in results_1d['top_fourier_dims']],
        'dominant_freqs': [int(f) for f in results_1d['dominant_freqs']],
        'top_sparsities': [float(x) for x in results_1d['top_sparsities']],
    }


def test_model_accuracy(model, tokenizer, p: int, n_samples: int = 100) -> float:
    """Test how well the model actually does modular arithmetic."""
    correct = 0
    
    np.random.seed(42)
    samples = [(np.random.randint(0, p), np.random.randint(0, p)) for _ in range(n_samples)]
    
    for a, b in tqdm(samples, desc="Testing accuracy"):
        prompt = create_prompt(a, b, p)
        expected = (a + b) % p
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract the number after "="
        try:
            answer_part = response.split("=")[-1].strip()
            predicted = int(answer_part.split()[0])
            if predicted == expected:
                correct += 1
        except (ValueError, IndexError):
            pass  # Failed to parse
    
    accuracy = correct / n_samples
    return accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--p", type=int, default=23, help="Prime modulus (keep small for speed)")
    parser.add_argument("--layers", type=str, default="5,10,15,20,25", 
                        help="Comma-separated layer indices to analyze")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--test_accuracy", action="store_true", help="Test model's arithmetic accuracy")
    parser.add_argument("--activation_type", type=str, default="hidden", choices=["mlp", "hidden"])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model, tokenizer = load_qwen3(args.model)
    n_layers = model.config.num_hidden_layers
    
    # Parse layer indices
    layers = [int(l) for l in args.layers.split(",")]
    layers = [l for l in layers if l < n_layers]
    
    print(f"\nPrime modulus: {args.p}")
    print(f"Total pairs to analyze: {args.p * args.p}")
    print(f"Layers to analyze: {layers}")
    
    # Test accuracy first
    if args.test_accuracy:
        print("\n" + "="*60)
        print("Testing model's modular arithmetic accuracy...")
        accuracy = test_model_accuracy(model, tokenizer, args.p)
        print(f"Accuracy on mod {args.p}: {accuracy:.2%}")
    
    # Analyze each layer
    all_results = []
    
    for layer_idx in layers:
        results = analyze_layer(
            model, tokenizer, args.p, layer_idx, output_dir, args.activation_type
        )
        all_results.append(results)
    
    # Save summary
    summary = {
        'model': args.model,
        'prime': args.p,
        'layers_analyzed': layers,
        'activation_type': args.activation_type,
        'results': all_results,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Layer':<10} {'Sig. Freqs':<15} {'Gini':<10} {'Top Freq':<10}")
    print("-"*50)
    for r in all_results:
        print(f"{r['layer']:<10} {r['n_significant_2d']:<15} {r['gini_2d']:<10.4f} {r['dominant_freqs'][0]:<10}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
