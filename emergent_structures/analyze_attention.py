import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_attention(model_id="Qwen/Qwen3-0.6B"):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        output_attentions=True
    )
    
    prompt = "2 + 3 mod 7 ="
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # outputs.attentions is a tuple of length num_layers
    # Each element is [batch, heads, seq_len, seq_len]
    attentions = outputs.attentions
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # We'll look at the "Fourier Layers" we identified: 7, 9, 14
    target_layers = [7, 9, 14]
    
    for layer_idx in target_layers:
        # Average over heads
        layer_attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(layer_attn, xticklabels=tokens, yticklabels=tokens, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Average Attention Layer {layer_idx}\nPrompt: '{prompt}'")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        
        output_dir = Path("emergent_structures/attention_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"attention_layer_{layer_idx}.png")
        plt.close()
        
        # Search for modulus token "7"
        mod_val = "7"
        try:
            # Match tokens that contain the string "7"
            mod_indices = [i for i, t in enumerate(tokens) if mod_val in t.replace('Ġ', '')]
            if mod_indices:
                mod_idx = mod_indices[0]
                avg_to_mod = layer_attn[-1, mod_idx]
                print(f"Layer {layer_idx}: Attention to modulus ('{tokens[mod_idx]}') from '=': {avg_to_mod:.4f}")
                
                # Print attention distribution for the last token '='
                print(f"Layer {layer_idx}: Attention distribution from '=':")
                for i, score in enumerate(layer_attn[-1]):
                    print(f"  -> {tokens[i]:<10}: {score:.4f}")
            else:
                print(f"Layer {layer_idx}: Could not find any token matching '{mod_val}' in {tokens}")
        except Exception as e:
            print(f"Layer {layer_idx}: Error during token search: {e}")

if __name__ == "__main__":
    analyze_attention()
