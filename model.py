"""
Small transformer for modular addition - replicating the grokking paper.

Reference: "Progress Measures for Grokking via Mechanistic Interpretability"
Nanda et al., ICLR 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum


class OneLayerTransformer(nn.Module):
    """
    One-layer transformer for modular addition.
    
    Architecture (following the paper):
    - Token embeddings (no tied embed/unembed)
    - Learned positional embeddings
    - 4 attention heads
    - ReLU MLP
    - No LayerNorm
    """
    
    def __init__(
        self,
        p: int = 53,  # Prime modulus
        d_model: int = 128,  # Embedding dimension
        n_heads: int = 4,  # Number of attention heads
        d_mlp: int = 512,  # MLP hidden dimension
    ):
        super().__init__()
        
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp
        
        # Vocabulary: p numbers + 1 special '=' token
        self.vocab_size = p + 1
        self.eq_token = p  # The '=' token index
        
        # Embeddings
        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions: a, b, =
        
        # Attention (combined QKV for efficiency)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / math.sqrt(d_model))
        
        # MLP
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        
        # Unembedding (not tied to embedding)
        self.unembed = nn.Linear(d_model, p, bias=False)
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            a: First number, shape (batch,)
            b: Second number, shape (batch,)
            
        Returns:
            logits: Shape (batch, p) - logits for each possible output
        """
        batch_size = a.shape[0]
        device = a.device
        
        # Create input sequence: [a, b, =]
        eq_tokens = torch.full((batch_size,), self.eq_token, device=device, dtype=torch.long)
        tokens = torch.stack([a, b, eq_tokens], dim=1)  # (batch, 3)
        
        # Token + positional embeddings
        positions = torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)  # (batch, 3, d_model)
        
        # Store for analysis
        self.cache = {'resid_pre_attn': x.clone()}
        
        # Attention
        # Queries, keys, values for all heads
        q = einsum(x, self.W_Q, 'b s d, h d k -> b h s k')  # (batch, heads, seq, d_head)
        k = einsum(x, self.W_K, 'b s d, h d k -> b h s k')
        v = einsum(x, self.W_V, 'b s d, h d k -> b h s k')
        
        # Attention scores
        attn_scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(self.d_head)
        
        # Causal mask (optional - the paper doesn't explicitly mention this)
        # For this task, we only read from position 2 (=), so masking isn't critical
        # but we'll include it for correctness
        mask = torch.triu(torch.ones(3, 3, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_pattern = F.softmax(attn_scores, dim=-1)
        self.cache['attn_pattern'] = attn_pattern.clone()
        
        # Apply attention
        attn_out = einsum(attn_pattern, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        attn_out = einsum(attn_out, self.W_O, 'b h s d, h d m -> b s m')
        
        # Residual connection
        x = x + attn_out
        self.cache['resid_post_attn'] = x.clone()
        
        # MLP (only on the final position, but we compute all for analysis)
        mlp_in = x
        mlp_hidden = F.relu(self.W_in(mlp_in))
        self.cache['mlp_activations'] = mlp_hidden.clone()
        
        mlp_out = self.W_out(mlp_hidden)
        
        # Residual connection
        x = x + mlp_out
        self.cache['resid_final'] = x.clone()
        
        # Read off logits from final position
        logits = self.unembed(x[:, 2, :])  # (batch, p)
        self.cache['logits'] = logits.clone()
        
        return logits
    
    def get_neuron_logit_map(self) -> torch.Tensor:
        """
        Compute W_L = W_U @ W_out, the neuron-to-logit mapping.
        This is the key matrix for understanding the Fourier structure.
        """
        # W_out.weight: (d_model, d_mlp) - from Linear(d_mlp, d_model)
        # unembed.weight: (p, d_model)
        # We want W_L: (p, d_mlp) = unembed @ W_out
        W_L = self.unembed.weight @ self.W_out.weight  # (p, d_model) @ (d_model, d_mlp) = (p, d_mlp)
        return W_L


def create_modular_addition_data(p: int, device: torch.device):
    """
    Create all pairs (a, b) for modular addition mod p.
    
    Returns:
        a: (p*p,) tensor of first operands
        b: (p*p,) tensor of second operands  
        targets: (p*p,) tensor of (a + b) mod p
    """
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    targets = (a + b) % p
    return a, b, targets


def train_test_split(p: int, train_frac: float, device: torch.device, seed: int = 42):
    """
    Split data into train and test sets.
    
    Args:
        p: Prime modulus
        train_frac: Fraction of data for training
        device: torch device
        seed: Random seed
        
    Returns:
        train_data: (a_train, b_train, targets_train)
        test_data: (a_test, b_test, targets_test)
    """
    torch.manual_seed(seed)
    
    a, b, targets = create_modular_addition_data(p, device)
    n_total = p * p
    n_train = int(n_total * train_frac)
    
    # Random permutation
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    train_data = (a[train_idx], b[train_idx], targets[train_idx])
    test_data = (a[test_idx], b[test_idx], targets[test_idx])
    
    return train_data, test_data


if __name__ == "__main__":
    from device_utils import get_device
    
    # Quick test
    device = get_device()
    
    p = 53
    model = OneLayerTransformer(p=p).to(device)
    
    # Test forward pass
    a = torch.tensor([0, 1, 2], device=device)
    b = torch.tensor([0, 1, 2], device=device)
    logits = model(a, b)
    print(f"Input shapes: a={a.shape}, b={b.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
