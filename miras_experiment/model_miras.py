import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum

class SinusoidalModulusEncoding(nn.Module):
    """
    Continuous scalar encoding for the modulus prime p.
    Maps p to a sinusoidal vector of d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # We learn a linear projection to help the model scale the 'magnitude' of p
        self.proj = nn.Linear(1, d_model)

    def forward(self, p: torch.Tensor):
        # p is (batch,) 
        p = p.float().unsqueeze(-1) # (batch, 1)
        
        # We create a simple frequency set
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)).to(p.device)
        
        pe = torch.zeros(p.shape[0], self.d_model).to(p.device)
        pe[:, 0::2] = torch.sin(p * div_term)
        pe[:, 1::2] = torch.cos(p * div_term)
        
        # Add a projected linear component so the model knows the absolute magnitude too
        return pe + self.proj(p)

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

class UniversalFourierTransformer(nn.Module):
    """
    Zero-Seed Emergent MIRAS. 
    Information must flow via Attention directly from the Prime token.
    """
    def __init__(self, max_p: int = 150, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512, d_mem: int = 128):
        super().__init__()
        self.max_p = max_p
        
        # ONLY a and b (and =) are category embeddings
        # Position 0 (p) will use SinusoidalModulusEncoding
        self.token_embed = nn.Embedding(max_p + 2, d_model)
        self.p_embedder = SinusoidalModulusEncoding(d_model)
        self.pos_embed = nn.Embedding(4, d_model) # [p, a, b, =]
        
        # This block is just a 'Neural Embedding' for the modulus
        self.memory_block = TitansMemory(d_model, d_mem)
        self.mem_to_hidden = nn.Linear(d_mem, d_model)
        
        self.layer1 = TransformerLayer(d_model, n_heads, d_mlp)
        self.layer2 = TransformerLayer(d_model, n_heads, d_mlp)
        self.unembed = nn.Linear(d_model, max_p)

    def forward(self, p_val: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        batch_size = p_val.shape[0]
        device = p_val.device
        eq_token = self.max_p + 1
        
        # 1. Embed special tokens (a, b, =)
        # Note: Position 0 is 'p' but we will overwrite it
        dummy_tokens = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
        dummy_tokens[:, 1] = a
        dummy_tokens[:, 2] = b
        dummy_tokens[:, 3] = eq_token
        
        x = self.token_embed(dummy_tokens) + self.pos_embed(torch.arange(4, device=device))
        
        # 2. Inject Sinusoidal Prime Encoding at position 0
        p_enc = self.p_embedder(p_val)
        x[:, 0, :] = p_enc
        
        # 3. Update the representation of token 'p' via Neural Memory
        z_p = self.memory_block(x[:, 0, :])
        x[:, 0, :] = self.mem_to_hidden(z_p)
        
        # 4. Two Transformer layers
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 5. Predict from the final '=' position
        return self.unembed(x[:, 3, :])
