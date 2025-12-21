import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum

class TitansMemory(nn.Module):
    """
    A simplified 'Titans' Memory Layer.
    It performs a 'Test-Time Update' (TTU) based on the surprise factor
    of the current token (the modulus p).
    """
    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        # Permanent weights (the 'Long-Term Memory' template)
        self.W_mem = nn.Parameter(torch.randn(d_model, d_mem) / math.sqrt(d_model))
        self.b_mem = nn.Parameter(torch.zeros(d_mem))
        
        # Surprise detection (predicts the current token from memory)
        self.surprise_net = nn.Linear(d_mem, d_model)

    def forward(self, x_p: torch.Tensor):
        """
        x_p: Embedding of the modulus token (B, D)
        Returns: Updated memory state vector (B, d_mem)
        """
        # 1. Initial lookup
        z = F.silu(x_p @ self.W_mem + self.b_mem) # (B, d_mem)
        
        # 2. Calculate 'Surprise'
        # In a real Titans model, we'd do a gradient step here.
        # To keep it differentiable in a single pass for this toy trainer:
        # We'll use a 'Self-Correction' gate which mimics a gradient update.
        # z_new = z - alpha * grad(Loss)
        # where Loss = ||surprise_net(z) - x_p||^2
        pred_x = self.surprise_net(z)
        surprise_signal = x_p - pred_x # The 'gradient' of the L2 loss w.r.t x_p
        
        # Map the surprise back into memory space to 'update' the state
        # This is the 'Test-Time Update' step
        update = F.silu(surprise_signal @ self.W_mem)
        z_updated = z + update # The 'shifted' equilibrium
        
        return z_updated

class UniversalFourierTransformer(nn.Module):
    """
    A 2-layer transformer using Titans memory to solve modular addition
    universally across different primes.
    """
    def __init__(
        self,
        max_p: int = 150,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        d_mem: int = 64
    ):
        super().__init__()
        self.max_p = max_p
        
        # Embeddings
        self.token_embed = nn.Embedding(max_p + 2, d_model)
        self.pos_embed = nn.Embedding(5, d_model) # [p, a, b, eq]
        
        # MIRAS Memory Layer (Updated at test-time)
        self.memory = TitansMemory(d_model, d_mem)
        
        # Communication channel for the memory state
        self.mem_to_resid = nn.Linear(d_mem, d_model)
        
        # Main Transformer Transformer Logic
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_model//n_heads) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_model//n_heads) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_model//n_heads) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, d_model//n_heads, d_model) / math.sqrt(d_model))
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )
        
        self.unembed = nn.Linear(d_model, max_p)

    def forward(self, p_val: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        batch_size = p_val.shape[0]
        device = p_val.device
        
        # Sequence: [p_val, a, b, eq]
        eq_token = self.max_p + 1
        tokens = torch.stack([p_val, a, b, torch.full_like(a, eq_token)], dim=1) # (B, 4)
        
        positions = torch.arange(4, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions) # (B, 4, D)
        
        # 1. Update Memory at POS 0 (the modulus)
        # This is the 'Surprise' step
        mem_state = self.memory(x[:, 0, :]) # (B, d_mem)
        mem_broadcast = self.mem_to_resid(mem_state).unsqueeze(1) # (B, 1, D)
        
        # 2. Inject Memory into the Residual Stream
        # This acts as the 'Universal Constant' for this specific prompt
        x = x + mem_broadcast 
        
        # 3. Standard Transformer Block
        # Attention
        q = einsum(x, self.W_Q, 'b s d, h d k -> b h s k')
        k = einsum(x, self.W_K, 'b s d, h d k -> b h s k')
        v = einsum(x, self.W_V, 'b s d, h d k -> b h s k')
        attn_scores = einsum(q, k, 'b h s1 d, b h s2 d -> b h s1 s2') / math.sqrt(q.shape[-1])
        attn_pattern = F.softmax(attn_scores, dim=-1)
        attn_out = einsum(attn_pattern, v, 'b h s1 s2, b h s2 d -> b h s1 d')
        attn_out = einsum(attn_out, self.W_O, 'b h s d, h d m -> b s m')
        
        x = x + attn_out
        x = x + self.mlp(x)
        
        # 4. Predict answer at '=' position
        logits = self.unembed(x[:, 3, :])
        
        return logits
