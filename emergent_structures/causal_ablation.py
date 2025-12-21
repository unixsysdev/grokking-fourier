"""
Causal Ablation Experiment for Qwen3 Fourier Dimensions

This script verifies if specific dimensions in Qwen3 are causally responsible
for modular arithmetic performance by zeroing them out (ablating) and
measuring the impact on arithmetic vs general text generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path

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

class AblationHook:
    def __init__(self, target_dim, layer_idx):
        self.target_dim = target_dim
        self.layer_idx = layer_idx
        self.active = False

    def hook_fn(self, module, input, output):
        if self.active:
            # output is (batch, seq, hidden_size)
            # We zero out the specific dimension
            output_clone = output.clone()
            output_clone[:, :, self.target_dim] = 0
            return output_clone
        return output

def test_arithmetic(model, tokenizer, p=23, n_samples=50):
    correct = 0
    np.random.seed(42)
    samples = [(np.random.randint(0, p), np.random.randint(0, p)) for _ in range(n_samples)]
    
    for a, b in tqdm(samples, desc=f"Arith mod {p}", leave=False):
        prompt = f"{a} + {b} mod {p} ="
        expected = (a + b) % p
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            answer_part = response.split("=")[-1].strip()
            # Handle cases like "5\n" or "5 (five)"
            predicted = int(answer_part.split()[0].replace(',', ''))
            if predicted == expected:
                correct += 1
        except (ValueError, IndexError):
            pass
            
    return correct / n_samples

def test_general_text(model, tokenizer, n_samples=20):
    """Test perplexity on general text fragments."""
    text_samples = [
        "The capital of France is Paris.",
        "Deep learning is a subset of machine learning.",
        "The quick brown fox jumps over the lazy dog.",
        "To be, or not to be, that is the question.",
        "A neural network is a series of algorithms that endeavors to recognize underlying relationships.",
        "Python is a high-level, general-purpose programming language.",
        "The sun rises in the east and sets in the west.",
        "Water boils at one hundred degrees Celsius.",
        "Mount Everest is the highest mountain in the world.",
        "Alan Turing was a British mathematician and computer scientist."
    ]
    
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(text_samples[:n_samples], desc="Language test", leave=False):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            tokens = inputs["input_ids"].size(1)
            total_loss += loss.item() * tokens
            total_tokens += tokens
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return np.exp(avg_loss)  # Return perplexity

def run_ablation_study(model, tokenizer, target_layer, target_dim):
    print(f"\n--- Studying Hidden Dim {target_dim} in Layer {target_layer} ---")
    
    # Register hook
    # hooks are attached to the output of the whole block (hidden state)
    hook = AblationHook(target_dim, target_layer)
    handle = model.model.layers[target_layer].register_forward_hook(hook.hook_fn)
    
    try:
        # Baseline
        print("Running Baseline...")
        hook.active = False
        base_acc = test_arithmetic(model, tokenizer)
        base_ppl = test_general_text(model, tokenizer)
        print(f"Baseline: Arith Acc = {base_acc:.2%}, Text PPL = {base_ppl:.4f}")
        
        # Ablated
        print("Running Ablated...")
        hook.active = True
        ablated_acc = test_arithmetic(model, tokenizer)
        ablated_ppl = test_general_text(model, tokenizer)
        print(f"Ablated:  Arith Acc = {ablated_acc:.2%}, Text PPL = {ablated_ppl:.4f}")
        
        acc_drop = (base_acc - ablated_acc)
        ppl_increase = (ablated_ppl - base_ppl) / base_ppl
        
        return {
            "dim": target_dim,
            "layer": target_layer,
            "base_acc": base_acc,
            "ablated_acc": ablated_acc,
            "acc_drop": acc_drop,
            "base_ppl": base_ppl,
            "ablated_ppl": ablated_ppl,
            "ppl_increase_pct": ppl_increase * 100
        }
    finally:
        handle.remove()

def main():
    model, tokenizer = load_model()
    
    # Targets based on previous analysis
    targets = [
        {"dim": 35, "layer": 9},   # Fundamental
        {"dim": 867, "layer": 14}, # Range detector
        {"dim": 8, "layer": 7},    # High R2 secondary
        {"dim": 100, "layer": 14}, # Random control dim
        {"dim": 500, "layer": 9},  # Random control dim
    ]
    
    results = []
    for t in targets:
        res = run_ablation_study(model, tokenizer, t["layer"], t["dim"])
        results.append(res)
        
    # Save results
    output_path = Path("ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSummary of Causal Impact:")
    print(f"{'Dim':<8} {'Layer':<8} {'Acc Drop':<12} {'PPL Inc %':<12}")
    print("-" * 45)
    for r in results:
        print(f"{r['dim']:<8} {r['layer']:<8} {r['acc_drop']:<12.2%} {r['ppl_increase_pct']:<12.2f}%")

if __name__ == "__main__":
    main()
