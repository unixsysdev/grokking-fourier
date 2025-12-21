"""
Improved SinPE Model for Better High-p Coverage

Key changes from model_miras.py:
1. Log-scaled encoding: p=120→121 has similar Δ as p=10→11
2. Higher frequencies: better resolution for large p
3. Multi-scale encoding: combines linear and log-scaled components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum


class ImprovedSinusoidalModulusEncoding(nn.Module):
    """
    Improved continuous scalar encoding for the modulus p.
    
    Changes from original:
    1. Log-scaled frequencies for uniform resolution across p range
    2. Higher base frequencies for better large-p discrimination
    3. Multi-scale: combines log(p) and linear(p) encodings
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Split dimensions: half for log-scale, half for linear-scale
        self.d_log = d_model // 2
        self.d_lin = d_model - self.d_log
        
        # Learned projection for magnitude information
        self.proj = nn.Linear(2, d_model)  # Takes [p, log(p)]
        
    def forward(self, p: torch.Tensor):
        # p is (batch,)
        p_float = p.float().unsqueeze(-1)  # (batch, 1)
        
        device = p.device
        
        # === Log-scaled encoding ===
        # Use log(p) so that p=120→121 has similar delta as p=10→11
        log_p = torch.log(p_float + 1)  # +1 to handle p=0
        
        # Higher frequencies for log scale (more resolution)
        log_freq = torch.exp(torch.arange(0, self.d_log, 2).float() * -(math.log(1000.0) / self.d_log)).to(device)
        
        pe_log = torch.zeros(p_float.shape[0], self.d_log, device=device)
        pe_log[:, 0::2] = torch.sin(log_p * log_freq * 10)  # Scale up for more oscillations
        pe_log[:, 1::2] = torch.cos(log_p * log_freq * 10)
        
        # === Linear-scaled encoding (original style but with higher frequencies) ===
        # Use higher base frequency (100 instead of 10000) for more oscillations
        lin_freq = torch.exp(torch.arange(0, self.d_lin, 2).float() * -(math.log(100.0) / self.d_lin)).to(device)
        
        pe_lin = torch.zeros(p_float.shape[0], self.d_lin, device=device)
        pe_lin[:, 0::2] = torch.sin(p_float * lin_freq)
        pe_lin[:, 1::2] = torch.cos(p_float * lin_freq)
        
        # Concatenate both encodings
        pe = torch.cat([pe_log, pe_lin], dim=-1)
        
        # Add learned linear component with both p and log(p)
        linear_input = torch.cat([p_float / 200.0, log_p / 6.0], dim=-1)  # Normalize
        
        return pe + self.proj(linear_input)


class TitansMemory(nn.Module):
    """
    Surprise-Driven Memory Layer (Titans).
    Encodes the prime modulus context into a state vector.
    """
    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.W_mem = nn.Parameter(torch.randn(d_model, d_mem) / math.sqrt(d_model))
        self.b_mem = nn.Parameter(torch.zeros(d_mem))
        self.surprise_net = nn.Linear(d_mem, d_model)

    def forward(self, x_p: torch.Tensor):
        z = F.silu(x_p @ self.W_mem + self.b_mem)
        pred_x = self.surprise_net(z)
        surprise_signal = x_p - pred_x
        update = F.silu(surprise_signal @ self.W_mem)
        return z + update


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / math.sqrt(d_model))
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # Attention
        norm_x = self.ln1(x)
        q = einsum(norm_x, self.W_Q, 'b s d, h d k -> b h s k')
        k = einsum(norm_x, self.W_K, 'b s d, h d k -> b h s k')
        v = einsum(norm_x, self.W_V, 'b s d, h d k -> b h s k')
        
        scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = einsum(attn, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        out = einsum(out, self.W_O, 'b h s d, h d m -> b s m')
        x = x + out
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x


class SparseUniversalTransformer(nn.Module):
    """
    Transformer with improved SinPE for sparse training across full modulus range.
    
    Key differences from UniversalFourierTransformer:
    - Uses ImprovedSinusoidalModulusEncoding (log-scaled, higher freq)
    - Same model size as original (d_model=256) for fair comparison
    """
    def __init__(self, max_p: int = 250, d_model: int = 256, n_heads: int = 8, d_mlp: int = 1024, d_mem: int = 256):
        super().__init__()
        self.max_p = max_p
        
        # Token embeddings for a and b (and =)
        self.token_embed = nn.Embedding(max_p + 2, d_model)
        
        # Improved SinPE for modulus encoding
        self.p_embedder = ImprovedSinusoidalModulusEncoding(d_model)
        
        # Positional embeddings for [p, a, b, =]
        self.pos_embed = nn.Embedding(4, d_model)
        
        # Memory block for processing modulus
        self.memory_block = TitansMemory(d_model, d_mem)
        self.mem_to_hidden = nn.Linear(d_mem, d_model)
        
        # Transformer layers
        self.layer1 = TransformerLayer(d_model, n_heads, d_mlp)
        self.layer2 = TransformerLayer(d_model, n_heads, d_mlp)
        
        # Output
        self.unembed = nn.Linear(d_model, max_p)

    def forward(self, p_val: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        batch_size = p_val.shape[0]
        device = p_val.device
        eq_token = self.max_p + 1
        
        # 1. Embed tokens (a, b, =)
        tokens_abc = torch.stack([a, b, torch.full_like(a, eq_token)], dim=1)
        x_abc = self.token_embed(tokens_abc) + self.pos_embed(torch.arange(1, 4, device=device))
        
        # 2. Improved SinPE for prime/modulus encoding at position 0
        p_enc = self.p_embedder(p_val).unsqueeze(1)  # (batch, 1, d_model)
        p_enc = p_enc + self.pos_embed(torch.zeros(1, dtype=torch.long, device=device))
        
        # Concatenate: [p, a, b, =]
        x = torch.cat([p_enc, x_abc], dim=1)  # (batch, 4, d_model)
        
        # 3. Process modulus through memory block
        z_p = self.memory_block(x[:, 0, :])
        p_hidden = self.mem_to_hidden(z_p).unsqueeze(1)
        x = torch.cat([p_hidden, x[:, 1:, :]], dim=1)
        
        # 4. Transformer layers
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 5. Predict from '=' position
        return self.unembed(x[:, 3, :])
