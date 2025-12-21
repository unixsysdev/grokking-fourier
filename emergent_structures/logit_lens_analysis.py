"""
Logit Lens Analysis for Qwen3 Fourier Representations

This script projects intermediate hidden states through the final LM Head
to see what "tokens" the model is thinking of at different layers
during modular arithmetic tasks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
import torch.nn.functional as F

def load_model(model_name="Qwen/Qwen3-0.6B"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def get_logit_lens(model, tokenizer, hidden_states, target_tokens):
    """
    Project hidden states onto the vocabulary.
    
    hidden_states: tuple of (num_layers + 1, batch, seq, hidden_size)
    target_tokens: list of token IDs we care about (e.g. 0 to p-1)
    
    Returns: probabilities of target tokens across all layers.
    Shape: (num_layers, len(target_tokens))
    """
    n_layers = len(hidden_states)
    probs_across_layers = []
    
    # We use the final layer norm and lm_head from the model
    ln_f = model.model.norm
    lm_head = model.lm_head
    
    for layer_idx in range(n_layers):
        # h shape: (batch, seq, hidden_size)
        h = hidden_states[layer_idx]
        
        # We only care about the last token position
        last_h = h[:, -1, :]  # (batch, hidden_size)
        
        with torch.no_grad():
            # Apply final layer norm
            norm_h = ln_f(last_h)
            # Apply LM Head to get logits
            logits = lm_head(norm_h)  # (batch, vocab_size)
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Extract probs for our target tokens
            target_probs = probs[0, target_tokens].cpu().numpy()
            probs_across_layers.append(target_probs)
            
    return np.array(probs_across_layers)

def run_analysis(model, tokenizer, p=23, sample_pairs=None):
    if sample_pairs is None:
        sample_pairs = [(5, 7), (10, 15), (20, 20)]
        
    # Get token IDs for digits '0' through '9'
    digit_tokens = [tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(10)]
    
    all_results = []
    
    for a, b in sample_pairs:
        # Include a trailing space to let the model "think" about the answer digit
        prompt = f"{a} + {b} mod {p} = "
        expected = (a + b) % p
        expected_str = str(expected)
        first_digit = int(expected_str[0])
        first_digit_token = digit_tokens[first_digit]
        
        print(f"\nAnalyzing: '{prompt}' (Expected: {expected}, First Digit: {first_digit})")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Get probs for digits 0-9
        probs = get_logit_lens(model, tokenizer, outputs.hidden_states, digit_tokens)
        
        fourier_acts = []
        for h in outputs.hidden_states:
            act = h[0, -1, 35].cpu().item()
            fourier_acts.append(act)
            
        all_results.append({
            "a": a, "b": b, "p": p, "expected": expected,
            "first_digit": first_digit,
            "probs": probs, # (layers, 10)
            "fourier_acts": fourier_acts
        })
        
    return all_results, digit_tokens

def plot_logit_lens(results, digit_tokens, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for res in results:
        a, b, p, expected, first_digit = res["a"], res["b"], res["p"], res["expected"], res["first_digit"]
        probs = res["probs"]
        fourier_acts = res["fourier_acts"]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        im = ax1.imshow(probs.T, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title(f"Logit Lens: Digit Prediction Probability for '{a} + {b} mod {p} = '")
        ax1.set_ylabel("Predicted First Digit (0-9)")
        ax1.set_xlabel("Layer Index")
        ax1.set_yticks(range(10))
        plt.colorbar(im, ax=ax1, label='Probability')
        
        ax1.axhline(first_digit, color='red', linestyle='--', alpha=0.5, label=f'True Digit ({first_digit})')
        ax1.legend()
        
        ax2.plot(fourier_acts, 'b-o', label='Dim 35 Activation')
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Activation (Dim 35)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"logit_lens_{a}_{b}_mod_{p}.png")
        plt.close()

def main():
    model, tokenizer = load_model()
    
    # We focus on a few interesting primes and pairs
    analyses = [
        (5, 7, 23),
        (15, 15, 23),
        (2, 3, 7)
    ]
    
    # Group by prime
    for p in [23, 7]:
        pairs = [(a, b) for a_curr, b_curr, p_curr in analyses if p_curr == p for a, b in [(a_curr, b_curr)]]
        if not pairs: continue
        
        results, number_tokens = run_analysis(model, tokenizer, p=p, sample_pairs=pairs)
        plot_logit_lens(results, number_tokens, "logit_lens_results")
        
        # Print top predictions for the first pair as a text sample
        res = results[0]
        print(f"\nTop 3 Predictions per Layer for {res['a']} + {res['b']} mod {res['p']}:")
        for layer_idx in [0, 5, 10, 15, 20, 25, 28]:
            if layer_idx >= len(res['probs']): continue
            p_layer = res['probs'][layer_idx]
            top_indices = np.argsort(p_layer)[-3:][::-1]
            top_nums = [i for i in top_indices]
            top_probs = [p_layer[i] for i in top_indices]
            print(f"Layer {layer_idx:2d}: " + ", ".join([f"{n} ({p:.1%})" for n, p in zip(top_nums, top_probs)]))

if __name__ == "__main__":
    main()
