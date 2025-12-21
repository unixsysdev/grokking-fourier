"""
Analyze Sparse Autoencoder (SAE) Features for Fourier Structure

Tests if the sparse features in the SAE are more "purely" Fourier
than the original model dimensions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
from train_sae import SparseAutoencoder

def fit_fourier_r2(activations, p):
    """Compute best Fourier fit R2 for a signal of length p."""
    # Organize by sum organized activations
    sum_activations = activations
    
    best_r2 = 0
    best_freq = 0
    
    for k in range(1, p//2 + 1):
        cos_wave = np.cos(2 * np.pi * k * np.arange(p) / p)
        sin_wave = np.sin(2 * np.pi * k * np.arange(p) / p)
        
        # Fit: a*cos + b*sin + c
        X = np.column_stack([cos_wave, sin_wave, np.ones(p)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, sum_activations, rcond=None)
            predicted = X @ coeffs
            ss_res = np.sum((sum_activations - predicted) ** 2)
            ss_tot = np.sum((sum_activations - np.mean(sum_activations)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            if r2 > best_r2:
                best_r2 = r2
                best_freq = k
        except:
            continue
            
    return best_r2, best_freq

def run_feature_analysis(model, tokenizer, sae, p=23):
    # Collect activations for ALL (a, b) pairs in mod p
    all_h = []
    for a in range(p):
        for b in range(p):
            prompt = f"{a} + {b} mod {p} ="
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[8][0, -1, :] # Layer 7 output
                all_h.append(hidden)
                
    all_h = torch.stack(all_h) # (p*p, 1024)
    
    # Pass through SAE encoder
    with torch.no_grad():
        _, z = sae(all_h.to(next(sae.parameters()).device))
        z = z.cpu().numpy() # (p*p, 4096)
        
    # Reorganize z by sum (a+b) mod p
    z_sum_organized = np.zeros((p, z.shape[1]))
    counts = np.zeros(p)
    
    idx = 0
    for a in range(p):
        for b in range(p):
            s = (a + b) % p
            z_sum_organized[s] += z[idx]
            counts[s] += 1
            idx += 1
            
    z_sum_organized /= counts[:, np.newaxis]
    
    # Find features with high Fourier R2
    feature_metrics = []
    for i in tqdm(range(z.shape[1]), desc="Analyzing SAE Features"):
        if np.max(z_sum_organized[:, i]) < 1e-4: # Skip dead neurons
            continue
            
        r2, freq = fit_fourier_r2(z_sum_organized[:, i], p)
        if r2 > 0.5:
            feature_metrics.append({
                "feature_idx": i,
                "r2": float(r2),
                "freq": int(freq),
                "max_act": float(np.max(z_sum_organized[:, i]))
            })
            
    # Sort by R2
    feature_metrics.sort(key=lambda x: x["r2"], reverse=True)
    return feature_metrics, z_sum_organized

def main():
    # 1. Load Model & SAE
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    sae_path = Path("sae_layer7.pt")
    if not sae_path.exists():
        print(f"Error: {sae_path} not found. Run train_sae.py first.")
        return
        
    sae = SparseAutoencoder(1024, 4096)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval().to(model.device)
    
    # 2. Analyze
    p = 23
    top_features, z_sum = run_feature_analysis(model, tokenizer, sae, p=p)
    
    # 3. Save Results
    with open("sae_analysis.json", "w") as f:
        json.dump(top_features, f, indent=2)
        
    print(f"\nTop 10 Fourier-like SAE Features (p={p}):")
    print(f"{'Index':<10} {'R2':<10} {'Freq':<10} {'Max Act':<10}")
    for f in top_features[:10]:
        print(f"{f['feature_idx']:<10} {f['r2']:<10.4f} {f['freq']:<10} {f['max_act']:<10.4f}")
        
    # 4. Plot top 4 features
    if top_features:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i in range(min(4, len(top_features))):
            f_idx = top_features[i]["feature_idx"]
            axes[i].plot(z_sum[:, f_idx], 'g-o')
            axes[i].set_title(f"SAE Feature {f_idx}\nR2={top_features[i]['r2']:.4f}, Freq={top_features[i]['freq']}")
            axes[i].set_xlabel("(a+b) mod p")
            
        plt.tight_layout()
        plt.savefig("sae_fourier_features.png")
        plt.close()
        print("\nPlot saved to emergent_structures/sae_fourier_features.png")

if __name__ == "__main__":
    main()
