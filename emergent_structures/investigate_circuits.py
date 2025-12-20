"""
Deep investigation of dimension 867 and the modular arithmetic circuits.

Experiments:
1. Test dim 867 on more numbers (primes and composites) to find its pattern
2. Analyze attention patterns to see how the model routes to different dims
3. Test composite numbers (mod 10, 12, 24) which appear often in training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def load_qwen3(model_name: str = "Qwen/Qwen3-0.6B", attn_implementation: str = "eager"):
    """Load Qwen3 model and tokenizer."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation=attn_implementation,  # Use eager for attention extraction
    )
    model.eval()
    
    return model, tokenizer


def extract_hidden_and_attention(model, tokenizer, prompt: str, layer_idx: int = 14):
    """Extract hidden state AND attention weights from a specific layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
    attention = outputs.attentions[layer_idx][0].cpu().numpy()  # (n_heads, seq_len, seq_len)
    
    return hidden, attention, inputs['input_ids'][0]


def analyze_dimension_for_modulus(model, tokenizer, n: int, dim_idx: int, layer_idx: int = 14):
    """
    Analyze how well a specific dimension fits cos(2πk(a+b)/n).
    Works for any modulus n (prime or composite).
    """
    activations = np.zeros((n, n))
    
    for a in range(n):
        for b in range(n):
            prompt = f"{a} + {b} mod {n} ="
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
            activations[a, b] = hidden[dim_idx]
    
    # Organize by sum
    sum_activations = np.zeros(n)
    for a in range(n):
        for b in range(n):
            s = (a + b) % n
            sum_activations[s] += activations[a, b]
    sum_activations /= n  # Each sum value appears n times
    
    # Fit cosines
    best_r2 = 0
    best_freq = 0
    
    for k in range(1, n // 2 + 1):
        cos_wave = np.cos(2 * np.pi * k * np.arange(n) / n)
        sin_wave = np.sin(2 * np.pi * k * np.arange(n) / n)
        
        X = np.column_stack([cos_wave, sin_wave, np.ones(n)])
        coeffs, _, _, _ = np.linalg.lstsq(X, sum_activations, rcond=None)
        
        predicted = X @ coeffs
        ss_res = np.sum((sum_activations - predicted) ** 2)
        ss_tot = np.sum((sum_activations - np.mean(sum_activations)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
        
        if r2 > best_r2:
            best_r2 = r2
            best_freq = k
    
    return best_r2, best_freq, sum_activations


def find_top_dims_for_modulus(model, tokenizer, n: int, layer_idx: int = 14, top_k: int = 10):
    """Find best Fourier dimensions for a given modulus."""
    hidden_size = model.config.hidden_size
    
    # Collect all activations
    activations = np.zeros((n, n, hidden_size))
    
    for a in tqdm(range(n), desc=f"mod {n}"):
        for b in range(n):
            prompt = f"{a} + {b} mod {n} ="
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
            activations[a, b, :] = hidden
    
    # Analyze each dimension
    results = []
    
    for d in range(hidden_size):
        act_2d = activations[:, :, d]
        
        # Organize by sum
        sum_act = np.zeros(n)
        for a in range(n):
            for b in range(n):
                s = (a + b) % n
                sum_act[s] += act_2d[a, b]
        sum_act /= n
        
        # Fit cosines
        best_r2 = 0
        best_freq = 0
        
        for k in range(1, n // 2 + 1):
            cos_wave = np.cos(2 * np.pi * k * np.arange(n) / n)
            sin_wave = np.sin(2 * np.pi * k * np.arange(n) / n)
            
            X = np.column_stack([cos_wave, sin_wave, np.ones(n)])
            coeffs, _, _, _ = np.linalg.lstsq(X, sum_act, rcond=None)
            
            predicted = X @ coeffs
            ss_res = np.sum((sum_act - predicted) ** 2)
            ss_tot = np.sum((sum_act - np.mean(sum_act)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
            
            if r2 > best_r2:
                best_r2 = r2
                best_freq = k
        
        results.append({'dim': d, 'r2': best_r2, 'freq': best_freq})
    
    results.sort(key=lambda x: x['r2'], reverse=True)
    return results[:top_k]


# =============================================================================
# Experiment 1: Deep dive into Dim 867
# =============================================================================

def experiment_dim867(model, tokenizer, output_dir: Path):
    """
    Why does dim 867 handle primes 17-29 but not others?
    Test on more numbers to find the pattern.
    """
    print("\n" + "="*60)
    print("Experiment 1: What's special about Dimension 867?")
    print("="*60)
    
    # Test a range of numbers (primes and composites)
    test_numbers = list(range(5, 36))  # 5 to 35
    
    results = []
    
    for n in tqdm(test_numbers, desc="Testing dim 867"):
        r2, freq, sum_act = analyze_dimension_for_modulus(model, tokenizer, n, dim_idx=867)
        is_prime = all(n % i != 0 for i in range(2, int(n**0.5) + 1)) and n > 1
        
        results.append({
            'n': n,
            'is_prime': is_prime,
            'r2': r2,
            'freq': freq
        })
        print(f"  n={n:2d} ({'prime' if is_prime else 'comp ':}): R² = {r2:.4f}, freq = {freq}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: R² by number
    ax = axes[0]
    ns = [r['n'] for r in results]
    r2s = [r['r2'] for r in results]
    colors = ['red' if r['is_prime'] else 'blue' for r in results]
    
    bars = ax.bar(ns, r2s, color=colors, alpha=0.7)
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.9')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.set_xlabel('Modulus n')
    ax.set_ylabel('R² for Dimension 867')
    ax.set_title('Dim 867: R² by Modulus (red=prime, blue=composite)')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Plot 2: R² vs n (scatter)
    ax = axes[1]
    primes = [r for r in results if r['is_prime']]
    composites = [r for r in results if not r['is_prime']]
    
    ax.scatter([r['n'] for r in primes], [r['r2'] for r in primes], 
               c='red', s=100, label='Primes', alpha=0.7)
    ax.scatter([r['n'] for r in composites], [r['r2'] for r in composites], 
               c='blue', s=100, label='Composites', alpha=0.7)
    
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Modulus n')
    ax.set_ylabel('R² for Dimension 867')
    ax.set_title('Dim 867: Does it prefer primes or certain ranges?')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dim867_investigation.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'dim867_investigation.png'}")
    
    # Analysis
    prime_r2s = [r['r2'] for r in results if r['is_prime']]
    composite_r2s = [r['r2'] for r in results if not r['is_prime']]
    
    print(f"\nDim 867 Statistics:")
    print(f"  Primes avg R²:     {np.mean(prime_r2s):.4f}")
    print(f"  Composites avg R²: {np.mean(composite_r2s):.4f}")
    
    # Find the "sweet spot"
    high_r2 = [r for r in results if r['r2'] > 0.85]
    print(f"\n  Numbers with R² > 0.85: {[r['n'] for r in high_r2]}")
    
    return results


# =============================================================================
# Experiment 2: Attention Pattern Analysis
# =============================================================================

def experiment_attention_routing(model, tokenizer, output_dir: Path):
    """
    How does attention route different moduli to different dimensions?
    Compare attention patterns for different primes.
    """
    print("\n" + "="*60)
    print("Experiment 2: Attention Routing Analysis")
    print("="*60)
    
    # Test prompts with different moduli
    test_cases = [
        ("5 + 3 mod 7 =", 7),
        ("5 + 3 mod 11 =", 11),
        ("5 + 3 mod 17 =", 17),
        ("5 + 3 mod 23 =", 23),
        ("5 + 3 mod 31 =", 31),
    ]
    
    layer_idx = 14
    attention_patterns = []
    
    for prompt, p in test_cases:
        hidden, attention, tokens = extract_hidden_and_attention(model, tokenizer, prompt, layer_idx)
        
        # Decode tokens for labeling
        token_strs = [tokenizer.decode([t]) for t in tokens]
        
        attention_patterns.append({
            'prompt': prompt,
            'p': p,
            'attention': attention,  # (n_heads, seq_len, seq_len)
            'tokens': token_strs,
            'hidden': hidden
        })
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Tokens: {token_strs}")
        print(f"  Attention shape: {attention.shape}")
    
    # Plot attention patterns
    n_cases = len(test_cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(12, 3*n_cases))
    
    for idx, data in enumerate(attention_patterns):
        ax = axes[idx]
        
        # Average attention across heads, look at last token's attention
        avg_attention = data['attention'].mean(axis=0)  # (seq_len, seq_len)
        last_token_attention = avg_attention[-1, :]  # What does last token attend to?
        
        ax.bar(range(len(data['tokens'])), last_token_attention)
        ax.set_xticks(range(len(data['tokens'])))
        ax.set_xticklabels(data['tokens'], rotation=45, ha='right')
        ax.set_ylabel('Attention')
        ax.set_title(f"'{data['prompt']}' — Last token attends to:")
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_routing.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'attention_routing.png'}")
    
    # Analyze: Does attention to 'p' token correlate with which dim activates?
    print("\nAttention to modulus token:")
    for data in attention_patterns:
        # Find the modulus token (the number before '=')
        tokens = data['tokens']
        # The modulus is typically right before '='
        mod_idx = None
        for i, t in enumerate(tokens):
            if t.strip() == '=':
                mod_idx = i - 1
                break
        
        if mod_idx is not None:
            avg_attention = data['attention'].mean(axis=0)
            attention_to_mod = avg_attention[-1, mod_idx]
            print(f"  p={data['p']:2d}: attention to '{tokens[mod_idx]}' = {attention_to_mod:.4f}")
    
    # Compare hidden states for different moduli
    print("\nHidden state differences (key dimensions):")
    key_dims = [35, 112, 216, 505, 867, 463]
    
    print(f"{'Dim':<8}", end="")
    for data in attention_patterns:
        print(f"p={data['p']:<6}", end="")
    print()
    print("-" * 50)
    
    for dim in key_dims:
        print(f"{dim:<8}", end="")
        for data in attention_patterns:
            val = data['hidden'][dim]
            print(f"{val:<10.3f}", end="")
        print()
    
    return attention_patterns


# =============================================================================
# Experiment 3: Composite Numbers
# =============================================================================

def experiment_composites(model, tokenizer, output_dir: Path):
    """
    Test composite numbers that appear often in training data:
    - mod 10 (decimal system)
    - mod 12 (hours, months)
    - mod 24 (hours in day)
    - mod 60 (minutes, seconds)
    - mod 100 (percentages)
    """
    print("\n" + "="*60)
    print("Experiment 3: Composite Numbers from Real Life")
    print("="*60)
    
    composites = [10, 12, 24, 60, 100]
    
    all_results = {}
    
    for n in composites:
        print(f"\n--- Testing mod {n} ---")
        
        # Find top Fourier dimensions
        top_dims = find_top_dims_for_modulus(model, tokenizer, n, top_k=10)
        all_results[n] = top_dims
        
        print(f"\nTop 10 Fourier dimensions for mod {n}:")
        print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
        print("-" * 26)
        for d in top_dims:
            print(f"{d['dim']:<8} {d['r2']:<10.4f} {d['freq']:<8}")
    
    # Compare with days (7), hours (12), months (12)
    print("\n" + "="*60)
    print("Comparison: Do clock/calendar composites reuse cyclic dims?")
    print("="*60)
    
    # Known cyclic dimensions from earlier experiments
    known_cyclic = {
        'days (7)': 125,
        'hours (12)': 725,
        'months (12)': 410,
    }
    
    print(f"\nKnown cyclic dimensions:")
    for name, dim in known_cyclic.items():
        print(f"  {name}: dim {dim}")
    
    print(f"\nDo these dims appear in composite top-10?")
    for n, top_dims in all_results.items():
        top_dim_ids = [d['dim'] for d in top_dims]
        print(f"\n  mod {n}:")
        for name, dim in known_cyclic.items():
            if dim in top_dim_ids:
                rank = top_dim_ids.index(dim) + 1
                r2 = top_dims[rank-1]['r2']
                print(f"    {name} dim {dim}: rank {rank}, R²={r2:.4f}")
            else:
                print(f"    {name} dim {dim}: not in top 10")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, n in enumerate(composites):
        ax = axes[idx]
        top_dims = all_results[n]
        
        dims = [d['dim'] for d in top_dims]
        r2s = [d['r2'] for d in top_dims]
        
        colors = []
        for d in dims:
            if d == 725:  # hours
                colors.append('red')
            elif d == 410:  # months
                colors.append('green')
            elif d == 125:  # days
                colors.append('blue')
            elif d == 867:  # medium primes
                colors.append('purple')
            else:
                colors.append('gray')
        
        ax.barh(range(len(dims)), r2s, color=colors, alpha=0.7)
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims)
        ax.set_xlabel('R²')
        ax.set_ylabel('Dimension')
        ax.set_title(f'mod {n}: Top Fourier Dims')
        ax.set_xlim(0, 1)
        ax.axvline(0.9, color='green', linestyle='--', alpha=0.3)
    
    # Legend in last subplot
    ax = axes[-1]
    ax.barh([0], [0], color='red', label='Dim 725 (hours)')
    ax.barh([0], [0], color='green', label='Dim 410 (months)')
    ax.barh([0], [0], color='blue', label='Dim 125 (days)')
    ax.barh([0], [0], color='purple', label='Dim 867 (med primes)')
    ax.barh([0], [0], color='gray', label='Other')
    ax.legend(loc='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.set_title('Legend')
    
    plt.suptitle('Composite Numbers: Which Fourier Dimensions Activate?', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'composites_analysis.png', dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'composites_analysis.png'}")
    
    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="results_deep")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated: dim867,attention,composites,all")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model, tokenizer = load_qwen3(args.model)
    
    if args.experiments == "all":
        experiments = ["dim867", "attention", "composites"]
    else:
        experiments = [e.strip() for e in args.experiments.split(",")]
    
    all_results = {}
    
    if "dim867" in experiments:
        all_results['dim867'] = experiment_dim867(model, tokenizer, output_dir)
    
    if "attention" in experiments:
        all_results['attention'] = experiment_attention_routing(model, tokenizer, output_dir)
    
    if "composites" in experiments:
        all_results['composites'] = experiment_composites(model, tokenizer, output_dir)
    
    # Save results
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
    
    # Filter out non-serializable attention data
    save_results = {}
    for k, v in all_results.items():
        if k == 'attention':
            save_results[k] = 'See attention_routing.png'
        else:
            save_results[k] = convert(v)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
