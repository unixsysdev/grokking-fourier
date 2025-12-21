import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from model_miras import UniversalFourierTransformer
from einops import einsum
from pathlib import Path

def visualize_attention(model, p, a, b, device, output_path):
    model.eval()
    
    # Sequence: [p, a, b, =]
    tokens = torch.tensor([[p, a, b, model.max_p + 1]], device=device)
    
    with torch.no_grad():
        x = model.token_embed(tokens) + model.pos_embed(torch.arange(4, device=device))
        
        # 1. Update representation of token 'p'
        z_p = model.memory_block(x[:, 0, :])
        x_new = x.clone()
        x_new[:, 0, :] = model.mem_to_hidden(z_p)
        x = x_new
        
        layers = [model.layer1, model.layer2]
        n_heads = model.layer1.n_heads
        fig, axes = plt.subplots(len(layers), n_heads, figsize=(4 * n_heads, 4 * len(layers)))
        
        for i, layer in enumerate(layers):
            norm_x = layer.ln1(x)
            q = einsum(norm_x, layer.W_Q, 'b s d, h d k -> b h s k')
            k = einsum(norm_x, layer.W_K, 'b s d, h d k -> b h s k')
            
            scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(layer.d_head)
            attn = F.softmax(scores, dim=-1) # (B, H, S, S)
            
            for h in range(n_heads):
                ax = axes[i, h] if len(layers) > 1 else axes[h]
                sns.heatmap(attn[0, h].cpu().numpy(), ax=ax, annot=True, cmap="Blues", fmt=".2f",
                            xticklabels=['p', 'a', 'b', '='],
                            yticklabels=['p', 'a', 'b', '='])
                ax.set_title(f"L{i+1} H{h} (mod {p})")
            
            # Continue forward pass
            v = einsum(norm_x, layer.W_V, 'b s d, h d k -> b h s k')
            out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
            out = einsum(out, layer.W_O, 'b h s d, h d m -> b s m')
            x = x + out
            x = x + layer.mlp(layer.ln2(x))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Use most recent checkpoint available
    checkpoints = sorted(list(Path("miras_experiment/checkpoints").glob("emergent_e*.pt")), 
                         key=lambda x: int(x.stem.split('e')[1]) if x.stem.split('e')[1].isdigit() else 0)
    
    if checkpoints:
        cp_path = checkpoints[-1]
        print(f"Loading checkpoint: {cp_path}")
        model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
        model.load_state_dict(torch.load(cp_path, map_location=device))
        
        out_dir = Path("miras_experiment/analysis/attention")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Test a few primes
        visualize_attention(model, 7, 2, 3, device, out_dir / "zero_seed_mod7.png")
        visualize_attention(model, 31, 15, 12, device, out_dir / "zero_seed_mod31.png")
        print(f"Attention maps saved to {out_dir}")
    else:
        print("No checkpoints found yet.")
