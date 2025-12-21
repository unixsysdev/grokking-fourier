"""
Sparse Autoencoder (SAE) Training for Qwen3 Layer 7

Decomposes the 1024-dimensional hidden states into a sparser
4096-dimensional space to investigate feature superposition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
import random

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model=1024, d_sparse=4096):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_sparse
        
        # Encoder: h -> z
        self.encoder = nn.Linear(d_model, d_sparse)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sparse))
        
        # Decoder: z -> h_hat
        self.decoder = nn.Linear(d_sparse, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        
        # Norm weights for safety
        nn.init.orthogonal_(self.decoder.weight)
        
    def forward(self, x):
        # x: (batch, d_model)
        x_cent = x - self.decoder_bias
        
        # Latent activations (ReLU(W*x + b))
        z = torch.relu(self.encoder(x_cent) + self.encoder_bias)
        
        # Reconstruction
        x_hat = self.decoder(z) + self.decoder_bias
        
        return x_hat, z

def collect_activations(model, tokenizer, layer_idx=7, n_samples=1000):
    """Collect hidden states from a mix of math and text."""
    activations = []
    
    # 1. Modular arithmetic samples
    p = 23
    math_samples = []
    for _ in range(n_samples // 2):
        a = random.randint(0, p-1)
        b = random.randint(0, p-1)
        math_samples.append(f"{a} + {b} mod {p} =")
        
    # 2. General text samples
    text_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we work.",
        "Deep learning models require significant computational resources.",
        "Modular arithmetic is essential for cryptography.",
        "The Fourier transform is a mathematical tool used in signal processing.",
        "Neural networks learn representations through backpropagation.",
        "Sparse autoencoders can decompose overlapping features.",
        "Transfomer models use attention mechanisms to process sequences."
    ]
    text_samples = random.choices(text_corpus, k=n_samples // 2)
    
    all_samples = math_samples + text_samples
    random.shuffle(all_samples)
    
    for prompt in tqdm(all_samples, desc="Collecting Activations"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Take last token hidden state from target layer
            hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu()
            activations.append(hidden)
            
    return torch.stack(activations)

def train_sae(activations, d_model=1024, d_sparse=4096, l1_coeff=3e-4, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = torch.device("mps")
        
    sae = SparseAutoencoder(d_model, d_sparse).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    
    # Dataset
    data = activations.to(device)
    
    print(f"Training SAE on {device} with L1={l1_coeff}...")
    for epoch in range(epochs):
        perm = torch.randperm(len(data))
        epoch_mse = 0
        epoch_l1 = 0
        
        for i in range(0, len(data), batch_size):
            batch_idx = perm[i:i + batch_size]
            x = data[batch_idx]
            
            optimizer.zero_grad()
            x_hat, z = sae(x)
            
            mse_loss = nn.MSELoss()(x_hat, x)
            l1_loss = l1_coeff * z.abs().sum() / x.size(0)
            
            loss = mse_loss + l1_loss
            loss.backward()
            optimizer.step()
            
            # Constrain decoder weights to unit norm (prevents "shrinkage" hack)
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(sae.decoder.weight.data, dim=0)
            
            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_l0 = (z > 0).float().sum(dim=-1).mean().item()
            print(f"Epoch {epoch+1:3d}: MSE={epoch_mse/len(data):.6f}, L1={epoch_l1/len(data):.6f}, L0={avg_l0:.1f}")
            
    return sae

def main():
    # Load model
    print("Loading Qwen3-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # 1. Collect activations
    # We use a smaller set for a demonstration run
    acts = collect_activations(model, tokenizer, layer_idx=7, n_samples=500)
    
    # 2. Train SAE
    sae = train_sae(acts, epochs=100, l1_coeff=1e-3)
    
    # 3. Save SAE weights
    output_path = Path("sae_layer7.pt")
    torch.save(sae.state_dict(), output_path)
    print(f"SAE weights saved to {output_path}")
    
    # 4. Save metadata
    with open("sae_config.json", "w") as f:
        json.dump({
            "d_model": 1024,
            "d_sparse": 4096,
            "layer_idx": 7,
            "l1_coeff": 1e-3
        }, f, indent=2)

if __name__ == "__main__":
    main()
