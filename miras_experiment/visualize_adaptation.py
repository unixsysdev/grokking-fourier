import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model_miras import UniversalFourierTransformer
from einops import einsum
import math

def get_neuron_activations(model, p, device):
    """Get the activations of all neurons in Layer 2 MLP for all (a,b)."""
    model.eval()
    with torch.no_grad():
        a = torch.arange(p, device=device).repeat_interleave(p)
        b = torch.arange(p, device=device).repeat(p)
        p_v = torch.full_like(a, p)
        
        # 1. SinPE Injection
        eq_token = model.max_p + 1
        tokens_abc = torch.stack([a, b, torch.full_like(a, eq_token)], dim=1)
        x_abc = model.token_embed(tokens_abc) + model.pos_embed(torch.arange(1, 4, device=device))
        
        p_enc = model.p_embedder(p_v).unsqueeze(1)
        p_enc = p_enc + model.pos_embed(torch.zeros(1, dtype=torch.long, device=device))
        x = torch.cat([p_enc, x_abc], dim=1)
        
        # 2. MIRAS Update
        z_p = model.memory_block(x[:, 0, :])
        p_hidden = model.mem_to_hidden(z_p).unsqueeze(1)
        x = torch.cat([p_hidden, x[:, 1:, :]], dim=1)
        
        # 3. Layer 1
        x = model.layer1(x)
        
        # 4. Layer 2 Hidden State (intercepting after MLP linear1)
        norm_x = model.layer2.ln1(x)
        # Handle Attention
        q = einsum(norm_x, model.layer2.W_Q, 'b s d, h d k -> b h s k')
        k = einsum(norm_x, model.layer2.W_K, 'b s d, h d k -> b h s k')
        v = einsum(norm_x, model.layer2.W_V, 'b s d, h d k -> b h s k')
        scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(model.layer2.d_head)
        attn = torch.softmax(scores, dim=-1)
        out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        out = einsum(out, model.layer2.W_O, 'b h s d, h d m -> b s m')
        x_post_attn = x + out
        
        # MLP Intercept at '=' token (idx 3)
        norm_x2 = model.layer2.ln2(x_post_attn)
        hidden = torch.relu(model.layer2.mlp[0](norm_x2[:, 3, :]))
        
    return hidden.cpu().numpy() # (p*p, d_mlp)

def plot_neuron_cross_prime(model, primes, neuron_idx, device, output_dir):
    """Plot a single neuron's activation against (a+b)%p for multiple primes."""
    fig, axes = plt.subplots(1, len(primes), figsize=(5*len(primes), 4))
    if len(primes) == 1: axes = [axes]
    
    for i, p in enumerate(primes):
        acts = get_neuron_activations(model, p, device)[:, neuron_idx]
        a = np.repeat(np.arange(p), p)
        b = np.tile(np.arange(p), p)
        sums = (a + b) % p
        
        avg_acts = np.zeros(p)
        for s in range(p):
            avg_acts[s] = acts[sums == s].mean()
            
        axes[i].plot(np.arange(p), avg_acts, marker='o', linestyle='-', color='purple')
        axes[i].set_title(f"Prime p={p}")
        axes[i].set_xlabel("(a+b) % p")
        if i == 0: axes[i].set_ylabel("Activation")
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(f"Neuron {neuron_idx} adaptation across primes")
    plt.tight_layout()
    plt.savefig(output_dir / f"neuron_{neuron_idx}_adaptation.png")
    plt.close()

def find_top_fourier_overlap(model, primes, device):
    """Find which neurons are 'highly active' across these primes."""
    prime_top_neurons = {}
    for p in primes:
        acts = get_neuron_activations(model, p, device) # (p*p, d_mlp)
        importance = acts.std(axis=0)
        top = np.argsort(importance)[-100:][::-1]
        prime_top_neurons[p] = set(top.tolist())
        print(f"p={p}: Top 10 neurons: {top[:10]}")
    
    # Check intersection
    all_sets = list(prime_top_neurons.values())
    intersection = set.intersection(*all_sets)
    print(f"\nNeurons active across ALL primes {primes}: {intersection}")
    return prime_top_neurons

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UniversalFourierTransformer(max_p=150, d_model=128, d_mem=128).to(device)
    # Using the latest RL checkpoint
    model.load_state_dict(torch.load("miras_experiment/checkpoints/rl_sinpe/rl_e35000.pt", map_location=device))
    
    output_dir = Path("miras_experiment/analysis/mechanics_rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_primes = [11, 17, 23] # Primes the RL model has mastered (>95%)
    # Find which neurons are active
    prime_top = find_top_fourier_overlap(model, test_primes, device)
    
    # Take the top neuron from the first prime
    top_n = list(prime_top[test_primes[0]])[0]
    plot_neuron_cross_prime(model, test_primes, top_n, device, output_dir)
    
    # Also take one from the second prime
    top_n2 = list(prime_top[test_primes[1]])[0]
    plot_neuron_cross_prime(model, test_primes, top_n2, device, output_dir)
    
    # If there is an intersection, plot it!
    all_sets = list(prime_top.values())
    intersection = set.intersection(*all_sets)
    if intersection:
        inter_n = list(intersection)[0]
        plot_neuron_cross_prime(model, test_primes, inter_n, device, output_dir)

if __name__ == "__main__":
    # Add a dummy attn_manual to support the visualization script if needed
    # or just use the model's standard forward for the layer.
    main()
