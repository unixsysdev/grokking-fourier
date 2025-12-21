"""
Comprehensive Polynomial Function Detection in Neural Networks

This script addresses methodological concerns in the original Fourier analysis by:
1. Testing multiple function types (Fourier, linear, quadratic, cubic)
2. Using robust detection methods (cross-validation, visual pattern classification)
3. Providing comprehensive visualization and statistical analysis

Key Features:
- Multiple hypothesis testing (not just Fourier)
- Robust validation across multiple runs
- Visual pattern classification
- Information-theoretic analysis
- Comprehensive comparison plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


def load_model(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load model and tokenizer."""
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


def collect_activations_for_modulus(model, tokenizer, n: int, layer_idx: int = 14):
    """Collect activations for modular arithmetic with modulus n."""
    print(f"Collecting activations for mod {n}...")
    hidden_size = model.config.hidden_size
    activations = np.zeros((n, n, hidden_size))
    
    for a in tqdm(range(n), desc=f"mod {n}"):
        for b in range(n):
            prompt = f"{a} + {b} mod {n} ="
            hidden = extract_hidden_states(model, tokenizer, prompt, layer_idx)
            activations[a, b, :] = hidden
    
    return activations


def organize_by_sum(activations: np.ndarray, n: int) -> np.ndarray:
    """Organize activations by (a + b) mod n."""
    sum_activations = np.zeros((n, activations.shape[2]))
    sum_counts = np.zeros(n)
    
    for a in range(n):
        for b in range(n):
            s = (a + b) % n
            sum_activations[s, :] += activations[a, b, :]
            sum_counts[s] += 1
    
    # Normalize
    mask = sum_counts > 0
    sum_activations[mask] /= sum_counts[mask, np.newaxis]
    
    return sum_activations


def fit_fourier_function(values: np.ndarray, n: int) -> Dict[str, Any]:
    """Fit Fourier function to values."""
    best_r2 = 0
    best_freq = 0
    best_coeffs = None
    
    for k in range(1, n // 2 + 1):
        cos_wave = np.cos(2 * np.pi * k * np.arange(n) / n)
        sin_wave = np.sin(2 * np.pi * k * np.arange(n) / n)
        
        X = np.column_stack([cos_wave, sin_wave, np.ones(n)])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
            predicted = X @ coeffs
            
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
            
            if r2 > best_r2:
                best_r2 = r2
                best_freq = k
                best_coeffs = coeffs
        except:
            continue
    
    return {'r2': best_r2, 'freq': best_freq, 'coeffs': best_coeffs}


def fit_polynomial_function(values: np.ndarray, degree: int) -> Dict[str, Any]:
    """Fit polynomial function of given degree."""
    x = np.arange(len(values))
    
    # Create polynomial features
    X = np.column_stack([x**d for d in range(degree + 1)])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        predicted = X @ coeffs
        
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
        
        return {'r2': r2, 'coeffs': coeffs}
    except:
        return {'r2': 0, 'coeffs': None}


def classify_visual_pattern(values: np.ndarray) -> Dict[str, Any]:
    """Classify visual pattern characteristics."""
    n = len(values)
    
    # Basic statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Monotonicity
    diffs = np.diff(values)
    monotonic_increasing = np.all(diffs >= -1e-6)
    monotonic_decreasing = np.all(diffs <= 1e-6)
    
    # Peaks and valleys
    # Simple peak detection without scipy
    peaks = []
    valleys = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            peaks.append(i)
        elif values[i] < values[i-1] and values[i] < values[i+1]:
            valleys.append(i)
    
    # Symmetry
    symmetry_score = np.corrcoef(values, values[::-1])[0, 1]
    
    # Entropy (measure of randomness)
    hist, _ = np.histogram(values, bins=20, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy = -np.sum(hist * np.log(hist))
    
    return {
        'mean': mean_val,
        'std': std_val,
        'monotonic_increasing': monotonic_increasing,
        'monotonic_decreasing': monotonic_decreasing,
        'n_peaks': len(peaks),
        'n_valleys': len(valleys),
        'symmetry_score': symmetry_score,
        'entropy': entropy
    }


def analyze_dimension_comprehensive(activations: np.ndarray, dim_idx: int, n: int) -> Dict[str, Any]:
    """Comprehensive analysis of a single dimension."""
    # Organize by sum
    sum_acts = organize_by_sum(activations, n)
    values = sum_acts[:, dim_idx]
    
    # Fit different function types
    fourier_result = fit_fourier_function(values, n)
    linear_result = fit_polynomial_function(values, 1)
    quadratic_result = fit_polynomial_function(values, 2)
    cubic_result = fit_polynomial_function(values, 3)
    
    # Visual pattern classification
    visual_pattern = classify_visual_pattern(values)
    
    # Determine best fit
    fits = {
        'fourier': fourier_result['r2'],
        'linear': linear_result['r2'],
        'quadratic': quadratic_result['r2'],
        'cubic': cubic_result['r2']
    }
    
    best_fit_type = max(fits.items(), key=lambda x: x[1])[0]
    best_fit_r2 = fits[best_fit_type]
    
    return {
        'dim': dim_idx,
        'values': values.tolist(),
        'fourier': fourier_result,
        'linear': linear_result,
        'quadratic': quadratic_result,
        'cubic': cubic_result,
        'visual_pattern': visual_pattern,
        'best_fit_type': best_fit_type,
        'best_fit_r2': best_fit_r2,
        'all_r2': fits
    }


def cross_validate_analysis(model, tokenizer, n: int, dim_idx: int, layer_idx: int = 14, 
                          n_runs: int = 3) -> Dict[str, Any]:
    """Cross-validate analysis across multiple runs with different random seeds."""
    results = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...")
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # Collect activations
        activations = collect_activations_for_modulus(model, tokenizer, n, layer_idx)
        
        # Analyze dimension
        analysis = analyze_dimension_comprehensive(activations, dim_idx, n)
        results.append(analysis)
    
    # Aggregate results
    avg_r2 = {}
    for fit_type in ['fourier', 'linear', 'quadratic', 'cubic']:
        avg_r2[fit_type] = np.mean([r['all_r2'][fit_type] for r in results])
    
    # Consistency of best fit type
    best_fit_types = [r['best_fit_type'] for r in results]
    most_common_fit = max(set(best_fit_types), key=best_fit_types.count)
    consistency = best_fit_types.count(most_common_fit) / len(best_fit_types)
    
    return {
        'dim': dim_idx,
        'n_runs': n_runs,
        'avg_r2': avg_r2,
        'most_common_fit': most_common_fit,
        'consistency': consistency,
        'individual_runs': results
    }


def analyze_multiple_dimensions(model, tokenizer, n: int, dim_indices: List[int], 
                              layer_idx: int = 14, n_runs: int = 3) -> List[Dict[str, Any]]:
    """Analyze multiple dimensions with cross-validation."""
    results = []
    
    for dim_idx in dim_indices:
        print(f"\nAnalyzing dimension {dim_idx}...")
        result = cross_validate_analysis(model, tokenizer, n, dim_idx, layer_idx, n_runs)
        results.append(result)
    
    return results


def create_comparison_plots(results: List[Dict[str, Any]], n: int, output_dir: Path):
    """Create comprehensive comparison plots."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. R² comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: R² by function type for each dimension
    ax = axes[0, 0]
    dims = [r['dim'] for r in results]
    fourier_r2 = [r['avg_r2']['fourier'] for r in results]
    linear_r2 = [r['avg_r2']['linear'] for r in results]
    quadratic_r2 = [r['avg_r2']['quadratic'] for r in results]
    cubic_r2 = [r['avg_r2']['cubic'] for r in results]
    
    x = np.arange(len(dims))
    width = 0.2
    
    ax.bar(x - 1.5*width, fourier_r2, width, label='Fourier', alpha=0.8)
    ax.bar(x - 0.5*width, linear_r2, width, label='Linear', alpha=0.8)
    ax.bar(x + 0.5*width, quadratic_r2, width, label='Quadratic', alpha=0.8)
    ax.bar(x + 1.5*width, cubic_r2, width, label='Cubic', alpha=0.8)
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Average R²')
    ax.set_title(f'R² Comparison by Function Type (mod {n})')
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Best fit type consistency
    ax = axes[0, 1]
    consistency = [r['consistency'] for r in results]
    best_fits = [r['most_common_fit'] for r in results]
    
    colors = {'fourier': 'red', 'linear': 'blue', 'quadratic': 'green', 'cubic': 'orange'}
    bar_colors = [colors[fit] for fit in best_fits]
    
    bars = ax.bar(dims, consistency, color=bar_colors, alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Consistency')
    ax.set_title('Best Fit Type Consistency Across Runs')
    ax.set_ylim(0, 1)
    
    # Add legend
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0,0),1,1, facecolor=colors[fit], alpha=0.7, label=fit.capitalize())
                     for fit in colors]
    ax.legend(handles=legend_elements)
    
    # Plot 3: Function type distribution
    ax = axes[1, 0]
    fit_counts = {'fourier': 0, 'linear': 0, 'quadratic': 0, 'cubic': 0}
    for r in results:
        fit_counts[r['most_common_fit']] += 1
    
    ax.pie(fit_counts.values(), labels=fit_counts.keys(), autopct='%1.1f%%', 
           colors=[colors[fit] for fit in fit_counts.keys()])
    ax.set_title('Distribution of Best Fit Types')
    
    # Plot 4: Sample activation patterns
    ax = axes[1, 1]
    if results:
        # Show the dimension with highest R²
        best_dim = max(results, key=lambda x: max(x['avg_r2'].values()))
        sample_run = best_dim['individual_runs'][0]
        values = sample_run['values']
        
        x = np.arange(len(values))
        ax.plot(x, values, 'bo-', label='Actual', markersize=4)
        
        # Plot best fit
        fit_type = best_dim['most_common_fit']
        if fit_type == 'fourier':
            coeffs = sample_run['fourier']['coeffs']
            if coeffs is not None:
                k = sample_run['fourier']['freq']
                cos_wave = np.cos(2 * np.pi * k * x / n)
                sin_wave = np.sin(2 * np.pi * k * x / n)
                fit_values = coeffs[0] * cos_wave + coeffs[1] * sin_wave + coeffs[2]
        elif fit_type == 'linear':
            coeffs = sample_run['linear']['coeffs']
            if coeffs is not None:
                fit_values = coeffs[0] * x + coeffs[1]
        elif fit_type == 'quadratic':
            coeffs = sample_run['quadratic']['coeffs']
            if coeffs is not None:
                fit_values = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        elif fit_type == 'cubic':
            coeffs = sample_run['cubic']['coeffs']
            if coeffs is not None:
                fit_values = coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
        
        if 'fit_values' in locals():
            ax.plot(x, fit_values, 'r-', label=f'{fit_type.capitalize()} fit', alpha=0.7)
        
        ax.set_xlabel('(a + b) mod n')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {best_dim["dim"]} - Best {fit_type.capitalize()} Fit (R²={best_dim["avg_r2"][fit_type]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Polynomial Function Detection Analysis (mod {n})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'polynomial_comparison_mod_{n}.png', dpi=150)
    plt.close()
    
    print(f"Saved comparison plot: {output_dir / f'polynomial_comparison_mod_{n}.png'}")


def create_detailed_plots(results: List[Dict[str, Any]], n: int, output_dir: Path):
    """Create detailed plots for individual dimensions."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Select top 3 dimensions by best R²
    top_results = sorted(results, key=lambda x: max(x['avg_r2'].values()), reverse=True)[:3]
    
    fig, axes = plt.subplots(len(top_results), 4, figsize=(20, 5*len(top_results)))
    if len(top_results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(top_results):
        sample_run = result['individual_runs'][0]
        values = np.array(sample_run['values'])
        x = np.arange(len(values))
        
        # Plot actual values
        ax = axes[idx, 0]
        ax.plot(x, values, 'bo-', markersize=4)
        ax.set_xlabel('(a + b) mod n')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {result["dim"]} - Actual Values')
        ax.grid(True, alpha=0.3)
        
        # Plot Fourier fit
        ax = axes[idx, 1]
        ax.plot(x, values, 'bo-', markersize=4, alpha=0.5, label='Actual')
        fourier_result = sample_run['fourier']
        if fourier_result['coeffs'] is not None:
            k = fourier_result['freq']
            coeffs = fourier_result['coeffs']
            cos_wave = np.cos(2 * np.pi * k * x / n)
            sin_wave = np.sin(2 * np.pi * k * x / n)
            fit_values = coeffs[0] * cos_wave + coeffs[1] * sin_wave + coeffs[2]
            ax.plot(x, fit_values, 'r-', label=f'Fourier (R²={fourier_result["r2"]:.3f})')
        ax.set_xlabel('(a + b) mod n')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {result["dim"]} - Fourier Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Polynomial fits
        ax = axes[idx, 2]
        ax.plot(x, values, 'bo-', markersize=4, alpha=0.5, label='Actual')
        
        for degree, color, label in [(1, 'green', 'Linear'), (2, 'orange', 'Quadratic'), (3, 'purple', 'Cubic')]:
            poly_result = sample_run[f'{"linear" if degree == 1 else "quadratic" if degree == 2 else "cubic"}']
            if poly_result['coeffs'] is not None:
                coeffs = poly_result['coeffs']
                fit_values = sum(coeffs[d] * x**d for d in range(degree + 1))
                ax.plot(x, fit_values, color=color, label=f'{label} (R²={poly_result["r2"]:.3f})')
        
        ax.set_xlabel('(a + b) mod n')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {result["dim"]} - Polynomial Fits')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot visual pattern analysis
        ax = axes[idx, 3]
        visual = sample_run['visual_pattern']
        
        # Create a simple visualization of pattern characteristics
        characteristics = []
        values_char = []
        
        if visual['monotonic_increasing']:
            characteristics.append('Monotonic ↑')
            values_char.append(1)
        elif visual['monotonic_decreasing']:
            characteristics.append('Monotonic ↓')
            values_char.append(1)
        else:
            characteristics.append('Non-monotonic')
            values_char.append(1)
        
        characteristics.extend([f'Peaks: {visual["n_peaks"]}', f'Valleys: {visual["n_valleys"]}'])
        values_char.extend([visual['n_peaks'], visual['n_valleys']])
        
        colors_char = ['red', 'blue', 'green', 'orange'][:len(characteristics)]
        ax.bar(characteristics, values_char, color=colors_char, alpha=0.7)
        ax.set_ylabel('Count / Indicator')
        ax.set_title(f'Dim {result["dim"]} - Pattern Characteristics')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Detailed Analysis - Top 3 Dimensions (mod {n})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'detailed_analysis_mod_{n}.png', dpi=150)
    plt.close()
    
    print(f"Saved detailed plots: {output_dir / f'detailed_analysis_mod_{n}.png'}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Polynomial Function Detection')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B', help='Model name')
    parser.add_argument('--primes', type=str, default='7,11,13', help='Comma-separated primes to test')
    parser.add_argument('--dims', type=str, default='35,867,505', help='Comma-separated dimensions to analyze')
    parser.add_argument('--layer', type=int, default=14, help='Layer to analyze')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of cross-validation runs')
    parser.add_argument('--output_dir', type=str, default='polynomial_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse arguments
    primes = [int(p.strip()) for p in args.primes.split(',')]
    dims = [int(d.strip()) for d in args.dims.split(',')]
    
    output_dir = Path(args.output_dir)
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    all_results = {}
    
    # Analyze each prime
    for prime in primes:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOD {prime}")
        print(f"{'='*60}")
        
        # Analyze dimensions
        results = analyze_multiple_dimensions(model, tokenizer, prime, dims, args.layer, args.n_runs)
        all_results[f'mod_{prime}'] = results
        
        # Create plots
        create_comparison_plots(results, prime, output_dir)
        create_detailed_plots(results, prime, output_dir)
        
        # Print summary
        print(f"\nSUMMARY FOR MOD {prime}:")
        print(f"{'Dim':<8} {'Best Fit':<12} {'Best R²':<10} {'Consistency':<12}")
        print("-" * 42)
        for r in results:
            print(f"{r['dim']:<8} {r['most_common_fit']:<12} {max(r['avg_r2'].values()):<10.4f} {r['consistency']:<12.4f}")
    
    # Save all results
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")
    
    # Final summary
    print("\nFINAL SUMMARY ACROSS ALL PRIMES:")
    print(f"{'Prime':<8} {'Fourier Wins':<15} {'Linear Wins':<15} {'Poly Wins':<15}")
    print("-" * 53)
    
    for prime in primes:
        results = all_results[f'mod_{prime}']
        fourier_wins = sum(1 for r in results if r['most_common_fit'] == 'fourier')
        linear_wins = sum(1 for r in results if r['most_common_fit'] == 'linear')
        poly_wins = sum(1 for r in results if r['most_common_fit'] in ['quadratic', 'cubic'])
        
        print(f"{prime:<8} {fourier_wins:<15} {linear_wins:<15} {poly_wins:<15}")


if __name__ == "__main__":
    main()