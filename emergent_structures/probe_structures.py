"""
Probing for Emergent Structures in Qwen3

This script tests whether Qwen3 has developed other Fourier-like or
structured representations for various domains beyond modular arithmetic.

Experiments:
1. Days of week (cyclic, period 7)
2. Months of year (cyclic, period 12)
3. Clock hours (cyclic, period 12 or 24)
4. Alphabet position (linear/cyclic, period 26)
5. Word analogies (parallelogram structure)
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
    """Extract hidden state from a specific layer for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden state at last token position
    hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
    return hidden


def extract_all_layers(model, tokenizer, prompt: str):
    """Extract hidden states from all layers."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    return [h[0, -1, :].cpu().numpy() for h in outputs.hidden_states[1:]]


# =============================================================================
# Experiment 1: Days of the Week (Period 7)
# =============================================================================

def experiment_days_of_week(model, tokenizer, output_dir: Path):
    """
    Test if the model has cyclic representations for days of the week.
    
    Prompt: "Monday + 3 days = " -> Thursday
    We vary the starting day and offset, look for periodic structure.
    """
    print("\n" + "="*60)
    print("Experiment 1: Days of the Week")
    print("="*60)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    n_days = len(days)
    
    # Collect activations for all (start_day, offset) pairs
    # This gives us a 7x7 grid, similar to modular arithmetic
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # Use middle layer for initial analysis
    test_layer = 14
    activations = np.zeros((n_days, n_days, hidden_size))
    
    for i, start_day in enumerate(tqdm(days, desc="Days of week")):
        for offset in range(n_days):
            prompt = f"{start_day} + {offset} days = "
            hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
            activations[i, offset, :] = hidden
    
    # Analyze: for each dimension, check if it correlates with (i + offset) mod 7
    results = analyze_cyclic_structure(activations, n_days, "days")
    
    # Find top Fourier-like dimensions
    top_dims = sorted(results, key=lambda x: x['best_r2'], reverse=True)[:10]
    
    print(f"\nTop 10 most cyclic dimensions:")
    print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
    print("-" * 26)
    for d in top_dims:
        print(f"{d['dim']:<8} {d['best_r2']:<10.4f} {d['best_freq']:<8}")
    
    # Plot top dimensions
    plot_cyclic_results(activations, top_dims[:5], n_days, days, 
                        "Days of Week", output_dir / "days_of_week.png")
    
    # Deep analysis of best dimension across all layers
    if top_dims[0]['best_r2'] > 0.3:
        best_dim = top_dims[0]['dim']
        layer_analysis = analyze_dimension_across_layers(
            model, tokenizer, best_dim, 
            lambda i, j: f"{days[i]} + {j} days = ",
            n_days, n_days
        )
        plot_layer_analysis(layer_analysis, best_dim, "Days of Week", 
                           output_dir / "days_of_week_layers.png")
    
    return {'experiment': 'days_of_week', 'period': 7, 'top_dims': top_dims}


# =============================================================================
# Experiment 2: Months of the Year (Period 12)
# =============================================================================

def experiment_months(model, tokenizer, output_dir: Path):
    """
    Test if the model has cyclic representations for months.
    
    Prompt: "January + 5 months = " -> June
    """
    print("\n" + "="*60)
    print("Experiment 2: Months of the Year")
    print("="*60)
    
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    n_months = len(months)
    
    hidden_size = model.config.hidden_size
    test_layer = 14
    activations = np.zeros((n_months, n_months, hidden_size))
    
    for i, start_month in enumerate(tqdm(months, desc="Months")):
        for offset in range(n_months):
            prompt = f"{start_month} + {offset} months = "
            hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
            activations[i, offset, :] = hidden
    
    results = analyze_cyclic_structure(activations, n_months, "months")
    top_dims = sorted(results, key=lambda x: x['best_r2'], reverse=True)[:10]
    
    print(f"\nTop 10 most cyclic dimensions:")
    print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
    print("-" * 26)
    for d in top_dims:
        print(f"{d['dim']:<8} {d['best_r2']:<10.4f} {d['best_freq']:<8}")
    
    plot_cyclic_results(activations, top_dims[:5], n_months, months,
                        "Months of Year", output_dir / "months.png")
    
    if top_dims[0]['best_r2'] > 0.3:
        best_dim = top_dims[0]['dim']
        layer_analysis = analyze_dimension_across_layers(
            model, tokenizer, best_dim,
            lambda i, j: f"{months[i]} + {j} months = ",
            n_months, n_months
        )
        plot_layer_analysis(layer_analysis, best_dim, "Months",
                           output_dir / "months_layers.png")
    
    return {'experiment': 'months', 'period': 12, 'top_dims': top_dims}


# =============================================================================
# Experiment 3: Clock Hours (Period 12)
# =============================================================================

def experiment_clock_hours(model, tokenizer, output_dir: Path):
    """
    Test if the model has cyclic representations for clock time.
    
    Prompt: "3 o'clock + 5 hours = " -> 8 o'clock
    """
    print("\n" + "="*60)
    print("Experiment 3: Clock Hours")
    print("="*60)
    
    n_hours = 12
    hidden_size = model.config.hidden_size
    test_layer = 14
    activations = np.zeros((n_hours, n_hours, hidden_size))
    
    for start_hour in tqdm(range(1, n_hours + 1), desc="Clock hours"):
        for offset in range(n_hours):
            prompt = f"{start_hour} o'clock + {offset} hours = "
            hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
            activations[start_hour - 1, offset, :] = hidden
    
    results = analyze_cyclic_structure(activations, n_hours, "hours")
    top_dims = sorted(results, key=lambda x: x['best_r2'], reverse=True)[:10]
    
    print(f"\nTop 10 most cyclic dimensions:")
    print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
    print("-" * 26)
    for d in top_dims:
        print(f"{d['dim']:<8} {d['best_r2']:<10.4f} {d['best_freq']:<8}")
    
    hour_labels = [str(h) for h in range(1, 13)]
    plot_cyclic_results(activations, top_dims[:5], n_hours, hour_labels,
                        "Clock Hours", output_dir / "clock_hours.png")
    
    if top_dims[0]['best_r2'] > 0.3:
        best_dim = top_dims[0]['dim']
        layer_analysis = analyze_dimension_across_layers(
            model, tokenizer, best_dim,
            lambda i, j: f"{i+1} o'clock + {j} hours = ",
            n_hours, n_hours
        )
        plot_layer_analysis(layer_analysis, best_dim, "Clock Hours",
                           output_dir / "clock_hours_layers.png")
    
    return {'experiment': 'clock_hours', 'period': 12, 'top_dims': top_dims}


# =============================================================================
# Experiment 4: Alphabet Position (Period 26)
# =============================================================================

def experiment_alphabet(model, tokenizer, output_dir: Path):
    """
    Test if the model has cyclic/linear representations for alphabet positions.
    
    Prompt: "A + 3 letters = " -> D
    Also test: "The 5th letter of the alphabet is "
    """
    print("\n" + "="*60)
    print("Experiment 4: Alphabet Position")
    print("="*60)
    
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n_letters = len(alphabet)
    
    hidden_size = model.config.hidden_size
    test_layer = 14
    
    # Test 1: Letter + offset (cyclic)
    # Use only first 13 letters as starting point to keep it manageable
    n_start = 13
    activations = np.zeros((n_start, n_letters, hidden_size))
    
    for i in tqdm(range(n_start), desc="Alphabet"):
        for offset in range(n_letters):
            prompt = f"{alphabet[i]} + {offset} letters = "
            hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
            activations[i, offset, :] = hidden
    
    # Analyze with period 26
    results = analyze_cyclic_structure(activations, n_letters, "letters", n_start=n_start)
    top_dims = sorted(results, key=lambda x: x['best_r2'], reverse=True)[:10]
    
    print(f"\nTop 10 most cyclic dimensions:")
    print(f"{'Dim':<8} {'R²':<10} {'Freq':<8}")
    print("-" * 26)
    for d in top_dims:
        print(f"{d['dim']:<8} {d['best_r2']:<10.4f} {d['best_freq']:<8}")
    
    letter_labels = list(alphabet)
    plot_cyclic_results(activations, top_dims[:5], n_letters, letter_labels,
                        "Alphabet Position", output_dir / "alphabet.png",
                        n_start=n_start)
    
    # Test 2: Direct position encoding
    # "The Nth letter is"
    print("\nTesting direct position encoding...")
    position_activations = np.zeros((n_letters, hidden_size))
    
    for i in tqdm(range(n_letters), desc="Position"):
        ordinal = get_ordinal(i + 1)
        prompt = f"The {ordinal} letter of the alphabet is "
        hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
        position_activations[i, :] = hidden
    
    # Check for linear or periodic structure in position encoding
    position_results = analyze_linear_structure(position_activations, n_letters)
    top_linear = sorted(position_results, key=lambda x: x['r2'], reverse=True)[:5]
    
    print(f"\nTop 5 dimensions with linear position encoding:")
    print(f"{'Dim':<8} {'R² (linear)':<12}")
    print("-" * 20)
    for d in top_linear:
        print(f"{d['dim']:<8} {d['r2']:<12.4f}")
    
    plot_position_encoding(position_activations, top_linear, alphabet,
                          output_dir / "alphabet_position.png")
    
    return {'experiment': 'alphabet', 'period': 26, 
            'top_cyclic_dims': top_dims, 'top_linear_dims': top_linear}


def get_ordinal(n):
    """Convert number to ordinal string."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


# =============================================================================
# Experiment 5: Word Analogies (Parallelogram Structure)
# =============================================================================

def experiment_analogies(model, tokenizer, output_dir: Path):
    """
    Test if word embeddings show parallelogram structure for analogies.
    
    Classic test: king - man + woman ≈ queen
    
    We test multiple analogy pairs and check if the vector arithmetic works.
    """
    print("\n" + "="*60)
    print("Experiment 5: Word Analogies")
    print("="*60)
    
    # Analogy pairs: (A, B, C, D) where A:B :: C:D
    analogies = [
        # Gender
        ("king", "queen", "man", "woman"),
        ("brother", "sister", "boy", "girl"),
        ("father", "mother", "son", "daughter"),
        ("husband", "wife", "uncle", "aunt"),
        # Country-Capital
        ("France", "Paris", "Japan", "Tokyo"),
        ("Germany", "Berlin", "Italy", "Rome"),
        ("Spain", "Madrid", "China", "Beijing"),
        # Verb tense
        ("walk", "walked", "run", "ran"),
        ("see", "saw", "eat", "ate"),
        ("go", "went", "come", "came"),
        # Comparative
        ("big", "bigger", "small", "smaller"),
        ("good", "better", "bad", "worse"),
        # Profession
        ("doctor", "hospital", "teacher", "school"),
        ("chef", "kitchen", "pilot", "cockpit"),
    ]
    
    hidden_size = model.config.hidden_size
    test_layer = 14
    
    # Get embeddings for all words
    all_words = set()
    for a, b, c, d in analogies:
        all_words.update([a, b, c, d])
    
    word_embeddings = {}
    for word in tqdm(all_words, desc="Getting word embeddings"):
        prompt = f"The word '{word}' means"
        hidden = extract_hidden_states(model, tokenizer, prompt, test_layer)
        word_embeddings[word] = hidden
    
    # Test parallelogram structure: A - B + C ≈ D
    # Equivalently: A - B ≈ C - D
    results = []
    
    print("\nAnalogy test results:")
    print(f"{'Analogy':<40} {'Cosine Sim':<12} {'Rank':<8}")
    print("-" * 60)
    
    for a, b, c, d in analogies:
        # Compute A - B + C
        vec_a = word_embeddings[a]
        vec_b = word_embeddings[b]
        vec_c = word_embeddings[c]
        vec_d = word_embeddings[d]
        
        # Predicted vector for D
        predicted_d = vec_a - vec_b + vec_c
        
        # Cosine similarity with actual D
        cos_sim = np.dot(predicted_d, vec_d) / (np.linalg.norm(predicted_d) * np.linalg.norm(vec_d) + 1e-10)
        
        # Rank: how many other words is D more similar to than our prediction?
        similarities = []
        for word, emb in word_embeddings.items():
            sim = np.dot(predicted_d, emb) / (np.linalg.norm(predicted_d) * np.linalg.norm(emb) + 1e-10)
            similarities.append((word, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        rank = next(i for i, (w, _) in enumerate(similarities) if w == d) + 1
        
        analogy_str = f"{a}:{b}::{c}:{d}"
        print(f"{analogy_str:<40} {cos_sim:<12.4f} {rank:<8}")
        
        results.append({
            'analogy': (a, b, c, d),
            'cosine_similarity': float(cos_sim),
            'rank': rank,
            'top_5_predicted': similarities[:5]
        })
    
    # Visualize the parallelogram structure using PCA
    plot_analogy_structure(word_embeddings, analogies[:4], 
                          output_dir / "analogies.png")
    
    # Check which dimensions contribute most to analogy structure
    # Look at consistency of (A-B) vs (C-D) directions
    direction_analysis = analyze_analogy_directions(word_embeddings, analogies)
    
    return {'experiment': 'analogies', 'results': results, 
            'direction_analysis': direction_analysis}


# =============================================================================
# Analysis Helper Functions
# =============================================================================

def analyze_cyclic_structure(activations, period, name, n_start=None):
    """
    Analyze if activations show cyclic structure.
    
    For each dimension, group by (i + j) mod period and fit cosine.
    """
    if n_start is None:
        n_start = period
    
    n_offset = activations.shape[1]
    hidden_size = activations.shape[2]
    
    results = []
    
    for d in range(hidden_size):
        # Group activations by (i + j) mod period
        sum_activations = np.zeros(period)
        sum_counts = np.zeros(period)
        
        for i in range(n_start):
            for j in range(n_offset):
                s = (i + j) % period
                sum_activations[s] += activations[i, j, d]
                sum_counts[s] += 1
        
        # Avoid division by zero
        mask = sum_counts > 0
        sum_activations[mask] /= sum_counts[mask]
        
        # Fit cosines at different frequencies
        best_r2 = 0
        best_freq = 0
        
        for k in range(1, period // 2 + 1):
            cos_wave = np.cos(2 * np.pi * k * np.arange(period) / period)
            sin_wave = np.sin(2 * np.pi * k * np.arange(period) / period)
            
            X = np.column_stack([cos_wave, sin_wave, np.ones(period)])
            
            # Only fit on positions where we have data
            X_masked = X[mask]
            y_masked = sum_activations[mask]
            
            if len(y_masked) < 3:
                continue
                
            coeffs, _, _, _ = np.linalg.lstsq(X_masked, y_masked, rcond=None)
            predicted = X_masked @ coeffs
            
            ss_res = np.sum((y_masked - predicted) ** 2)
            ss_tot = np.sum((y_masked - np.mean(y_masked)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
            
            if r2 > best_r2:
                best_r2 = r2
                best_freq = k
        
        results.append({
            'dim': d,
            'best_r2': best_r2,
            'best_freq': best_freq,
            'sum_activations': sum_activations.tolist()
        })
    
    return results


def analyze_linear_structure(activations, n_positions):
    """Check if activations show linear encoding of position."""
    hidden_size = activations.shape[1]
    positions = np.arange(n_positions)
    
    results = []
    
    for d in range(hidden_size):
        values = activations[:, d]
        
        # Linear fit
        X = np.column_stack([positions, np.ones(n_positions)])
        coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
        predicted = X @ coeffs
        
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
        
        results.append({
            'dim': d,
            'r2': r2,
            'slope': coeffs[0],
            'intercept': coeffs[1]
        })
    
    return results


def analyze_analogy_directions(word_embeddings, analogies):
    """Analyze which dimensions are most consistent for analogy directions."""
    hidden_size = len(next(iter(word_embeddings.values())))
    
    # For each dimension, compute variance of (A-B) - (C-D) across analogies
    dim_consistency = []
    
    for d in range(hidden_size):
        diffs = []
        for a, b, c, d_word in analogies:
            diff_ab = word_embeddings[a][d] - word_embeddings[b][d]
            diff_cd = word_embeddings[c][d] - word_embeddings[d_word][d]
            diffs.append(diff_ab - diff_cd)
        
        # Lower variance = more consistent analogy structure
        variance = np.var(diffs)
        mean_diff = np.mean([abs(d) for d in diffs])
        
        dim_consistency.append({
            'dim': d,
            'variance': variance,
            'mean_abs_diff': mean_diff
        })
    
    return sorted(dim_consistency, key=lambda x: x['variance'])[:20]


def analyze_dimension_across_layers(model, tokenizer, dim_idx, prompt_fn, n_i, n_j):
    """Analyze a specific dimension across all layers."""
    n_layers = model.config.num_hidden_layers
    layer_r2 = []
    
    for layer in tqdm(range(n_layers), desc=f"Analyzing dim {dim_idx} across layers"):
        activations = np.zeros((n_i, n_j))
        
        for i in range(n_i):
            for j in range(n_j):
                prompt = prompt_fn(i, j)
                hidden = extract_hidden_states(model, tokenizer, prompt, layer)
                activations[i, j] = hidden[dim_idx]
        
        # Compute R² for this layer
        period = max(n_i, n_j)
        sum_act = np.zeros(period)
        sum_cnt = np.zeros(period)
        
        for i in range(n_i):
            for j in range(n_j):
                s = (i + j) % period
                sum_act[s] += activations[i, j]
                sum_cnt[s] += 1
        
        mask = sum_cnt > 0
        sum_act[mask] /= sum_cnt[mask]
        
        # Best cosine fit
        best_r2 = 0
        for k in range(1, period // 2 + 1):
            cos_wave = np.cos(2 * np.pi * k * np.arange(period) / period)
            sin_wave = np.sin(2 * np.pi * k * np.arange(period) / period)
            X = np.column_stack([cos_wave, sin_wave, np.ones(period)])
            X_m, y_m = X[mask], sum_act[mask]
            
            if len(y_m) < 3:
                continue
            
            coeffs, _, _, _ = np.linalg.lstsq(X_m, y_m, rcond=None)
            pred = X_m @ coeffs
            ss_res = np.sum((y_m - pred) ** 2)
            ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
            best_r2 = max(best_r2, r2)
        
        layer_r2.append(best_r2)
    
    return layer_r2


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_cyclic_results(activations, top_dims, period, labels, title, output_path, n_start=None):
    """Plot the top cyclic dimensions."""
    if n_start is None:
        n_start = period
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: 2D activation pattern for best dimension
    ax = axes[0, 0]
    best_dim = top_dims[0]['dim']
    im = ax.imshow(activations[:, :, best_dim], cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Offset')
    ax.set_ylabel('Start')
    ax.set_title(f'Dim {best_dim} Activations')
    plt.colorbar(im, ax=ax)
    
    # Plot 2-6: Sum-organized activations for top 5 dimensions
    for idx, dim_info in enumerate(top_dims[:5]):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        dim = dim_info['dim']
        
        # Compute sum-organized activations
        sum_act = np.zeros(period)
        sum_cnt = np.zeros(period)
        for i in range(n_start):
            for j in range(activations.shape[1]):
                s = (i + j) % period
                sum_act[s] += activations[i, j, dim]
                sum_cnt[s] += 1
        mask = sum_cnt > 0
        sum_act[mask] /= sum_cnt[mask]
        
        ax.plot(range(period), sum_act, 'bo-', markersize=4)
        ax.set_xlabel(f'(start + offset) mod {period}')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {dim} (R²={dim_info["best_r2"]:.3f}, k={dim_info["best_freq"]})')
        ax.grid(True, alpha=0.3)
        
        # Add cosine fit
        k = dim_info['best_freq']
        if k > 0:
            cos_wave = np.cos(2 * np.pi * k * np.arange(period) / period)
            sin_wave = np.sin(2 * np.pi * k * np.arange(period) / period)
            X = np.column_stack([cos_wave, sin_wave, np.ones(period)])
            X_m, y_m = X[mask], sum_act[mask]
            if len(y_m) >= 3:
                coeffs, _, _, _ = np.linalg.lstsq(X_m, y_m, rcond=None)
                fit = X @ coeffs
                ax.plot(range(period), fit, 'r-', alpha=0.7, label='Cosine fit')
    
    plt.suptitle(f'{title}: Cyclic Structure Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_analysis(layer_r2, dim_idx, title, output_path):
    """Plot R² across layers for a dimension."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    layers = range(len(layer_r2))
    ax.plot(layers, layer_r2, 'bo-')
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² (cosine fit)')
    ax.set_title(f'{title}: Dimension {dim_idx} Across Layers')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    
    best_layer = np.argmax(layer_r2)
    ax.annotate(f'Best: L{best_layer} (R²={layer_r2[best_layer]:.3f})',
                xy=(best_layer, layer_r2[best_layer]),
                xytext=(best_layer + 2, layer_r2[best_layer] + 0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_position_encoding(activations, top_dims, labels, output_path):
    """Plot linear position encoding."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    n_positions = len(labels)
    positions = np.arange(n_positions)
    
    for idx, dim_info in enumerate(top_dims[:6]):
        ax = axes[idx // 3, idx % 3]
        dim = dim_info['dim']
        
        values = activations[:, dim]
        ax.plot(positions, values, 'bo-', markersize=4)
        
        # Add linear fit
        slope, intercept = dim_info['slope'], dim_info['intercept']
        fit = slope * positions + intercept
        ax.plot(positions, fit, 'r-', alpha=0.7, label=f'Linear (R²={dim_info["r2"]:.3f})')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Activation')
        ax.set_title(f'Dim {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add letter labels
        ax.set_xticks(positions[::5])
        ax.set_xticklabels([labels[i] for i in range(0, n_positions, 5)])
    
    plt.suptitle('Alphabet: Linear Position Encoding', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_analogy_structure(word_embeddings, analogies, output_path):
    """Visualize analogy structure using PCA (manual implementation)."""
    # Manual PCA implementation to avoid sklearn dependency
    def manual_pca(X, n_components=2):
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        # Compute covariance matrix
        cov = np.cov(X_centered.T)
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Project
        components = eigenvectors[:, :n_components]
        projected = X_centered @ components
        explained_var = eigenvalues[:n_components] / eigenvalues.sum()
        return projected, explained_var
    
    # Get all words from the analogies
    words = []
    for a, b, c, d in analogies:
        words.extend([a, b, c, d])
    words = list(dict.fromkeys(words))  # Remove duplicates, preserve order
    
    # Stack embeddings
    embeddings = np.array([word_embeddings[w] for w in words])
    
    # PCA to 2D (manual)
    coords, explained_var = manual_pca(embeddings, n_components=2)
    word_coords = {w: coords[i] for i, w in enumerate(words)}
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analogies)))
    
    for idx, (a, b, c, d) in enumerate(analogies):
        color = colors[idx]
        
        # Plot points
        for word in [a, b, c, d]:
            x, y = word_coords[word]
            ax.scatter(x, y, c=[color], s=100, zorder=5)
            ax.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Draw parallelogram
        coords_a = word_coords[a]
        coords_b = word_coords[b]
        coords_c = word_coords[c]
        coords_d = word_coords[d]
        
        # A->B and C->D should be parallel
        ax.arrow(coords_a[0], coords_a[1], coords_b[0]-coords_a[0], coords_b[1]-coords_a[1],
                head_width=0.02, head_length=0.01, fc=color, ec=color, alpha=0.5)
        ax.arrow(coords_c[0], coords_c[1], coords_d[0]-coords_c[0], coords_d[1]-coords_c[1],
                head_width=0.02, head_length=0.01, fc=color, ec=color, alpha=0.5)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    ax.set_title('Word Analogies: Parallelogram Structure in PCA Space')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="emergent_structures")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated list of experiments: days,months,clock,alphabet,analogies,all")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    model, tokenizer = load_qwen3(args.model)
    
    # Parse experiments
    if args.experiments == "all":
        experiments = ["days", "months", "clock", "alphabet", "analogies"]
    else:
        experiments = [e.strip() for e in args.experiments.split(",")]
    
    all_results = {}
    
    # Run experiments
    if "days" in experiments:
        all_results['days_of_week'] = experiment_days_of_week(model, tokenizer, output_dir)
    
    if "months" in experiments:
        all_results['months'] = experiment_months(model, tokenizer, output_dir)
    
    if "clock" in experiments:
        all_results['clock_hours'] = experiment_clock_hours(model, tokenizer, output_dir)
    
    if "alphabet" in experiments:
        all_results['alphabet'] = experiment_alphabet(model, tokenizer, output_dir)
    
    if "analogies" in experiments:
        all_results['analogies'] = experiment_analogies(model, tokenizer, output_dir)
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        if 'top_dims' in results:
            best = results['top_dims'][0]
            print(f"  Best dimension: {best['dim']} with R² = {best['best_r2']:.4f}")
        if 'top_cyclic_dims' in results:
            best = results['top_cyclic_dims'][0]
            print(f"  Best cyclic dim: {best['dim']} with R² = {best['best_r2']:.4f}")
        if 'results' in results and isinstance(results['results'], list):
            # Analogies
            avg_rank = np.mean([r['rank'] for r in results['results']])
            avg_sim = np.mean([r['cosine_similarity'] for r in results['results']])
            print(f"  Average cosine sim: {avg_sim:.4f}, Average rank: {avg_rank:.1f}")
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
