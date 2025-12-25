"""
Addition Transformer Model

A small transformer that learns addition from character-level representation.
Input:  "123+456=" (as character tokens)
Output: "579" (generated digit-by-digit)

Key design choices:
1. Character-level tokenization (0-9, +, =, padding)
2. Encoder-decoder style: encode the equation, decode the answer
3. Small enough to train quickly, big enough to learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class AdditionTransformer(nn.Module):
    """
    Transformer for learning addition.
    
    Architecture:
    - Token embedding for characters (0-9, +, =, <pad>, <sos>, <eos>)
    - Positional encoding
    - Transformer encoder (processes "123+456=")
    - Transformer decoder (generates "579" autoregressively)
    """
    
    # Token vocabulary
    TOKENS = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        '+': 10, '=': 11,
        '<pad>': 12, '<sos>': 13, '<eos>': 14
    }
    INV_TOKENS = {v: k for k, v in TOKENS.items()}
    VOCAB_SIZE = 15
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.token_embed = nn.Embedding(self.VOCAB_SIZE, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.VOCAB_SIZE)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    @classmethod
    def tokenize(cls, s: str) -> list:
        """Convert string to token indices."""
        return [cls.TOKENS[c] for c in s]
    
    @classmethod
    def detokenize(cls, tokens: list) -> str:
        """Convert token indices back to string."""
        result = []
        for t in tokens:
            if t == cls.TOKENS['<eos>']:
                break
            if t == cls.TOKENS['<sos>'] or t == cls.TOKENS['<pad>']:
                continue
            result.append(cls.INV_TOKENS[t])
        return ''.join(result)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode the input equation.
        
        Args:
            src: (batch, src_len) token indices
            src_mask: (batch, src_len) padding mask (True = ignore)
        
        Returns:
            memory: (batch, src_len, d_model) encoded representation
        """
        # Embed and add positional encoding
        x = self.token_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Encode
        memory = self.encoder(x, src_key_padding_mask=src_mask)
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode to produce output logits.
        
        Args:
            tgt: (batch, tgt_len) target token indices
            memory: (batch, src_len, d_model) encoder output
            tgt_mask: (tgt_len, tgt_len) causal mask
            memory_mask: (batch, src_len) padding mask for encoder output
        
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        x = self.token_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)
        
        # Decode
        output = self.decoder(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        return logits
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Full forward pass for training.
        
        Args:
            src: (batch, src_len) input equation tokens
            tgt: (batch, tgt_len) target answer tokens (with <sos> prepended)
            src_mask: (batch, src_len) padding mask
            tgt_mask: (tgt_len, tgt_len) causal mask
        
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(tgt, memory, tgt_mask, src_mask)
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        max_len: int = 10
    ) -> torch.Tensor:
        """
        Generate answer autoregressively.
        
        Args:
            src: (batch, src_len) input equation tokens
            src_mask: (batch, src_len) padding mask
            max_len: maximum output length
        
        Returns:
            output: (batch, output_len) generated token indices
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode input
        memory = self.encode(src, src_mask)
        
        # Start with <sos>
        output = torch.full((batch_size, 1), self.TOKENS['<sos>'], dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Decode
            logits = self.decode(output, memory, memory_mask=src_mask)
            
            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)
            
            # Stop if all sequences have generated <eos>
            if (next_token == self.TOKENS['<eos>']).all():
                break
        
        return output


def create_addition_sample(max_digits: int = 5):
    """
    Create a single addition sample.
    
    Returns:
        equation: str like "123+456="
        answer: str like "579"
    """
    import random
    
    # Random number of digits for each operand (1 to max_digits)
    digits_a = random.randint(1, max_digits)
    digits_b = random.randint(1, max_digits)
    
    # Generate numbers
    a = random.randint(10**(digits_a-1) if digits_a > 1 else 0, 10**digits_a - 1)
    b = random.randint(10**(digits_b-1) if digits_b > 1 else 0, 10**digits_b - 1)
    
    # Create strings
    equation = f"{a}+{b}="
    answer = str(a + b)
    
    return equation, answer


def collate_batch(samples: list, device: torch.device):
    """
    Collate samples into batched tensors.
    
    Args:
        samples: list of (equation, answer) tuples
        device: torch device
    
    Returns:
        src: (batch, max_src_len) padded source tokens
        tgt_input: (batch, max_tgt_len) target with <sos> prepended
        tgt_output: (batch, max_tgt_len) target with <eos> appended
        src_mask: (batch, max_src_len) padding mask
    """
    equations, answers = zip(*samples)
    
    # Tokenize
    src_tokens = [AdditionTransformer.tokenize(eq) for eq in equations]
    tgt_tokens = [AdditionTransformer.tokenize(ans) for ans in answers]
    
    # Get max lengths
    max_src = max(len(s) for s in src_tokens)
    max_tgt = max(len(t) for t in tgt_tokens) + 1  # +1 for <sos> or <eos>
    
    # Pad
    pad_idx = AdditionTransformer.TOKENS['<pad>']
    sos_idx = AdditionTransformer.TOKENS['<sos>']
    eos_idx = AdditionTransformer.TOKENS['<eos>']
    
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    src_mask = []
    
    for s, t in zip(src_tokens, tgt_tokens):
        # Pad source
        src_padded.append(s + [pad_idx] * (max_src - len(s)))
        src_mask.append([False] * len(s) + [True] * (max_src - len(s)))
        
        # Target input: <sos> + answer
        tgt_in = [sos_idx] + t + [pad_idx] * (max_tgt - len(t) - 1)
        tgt_input_padded.append(tgt_in)
        
        # Target output: answer + <eos>
        tgt_out = t + [eos_idx] + [pad_idx] * (max_tgt - len(t) - 1)
        tgt_output_padded.append(tgt_out)
    
    return (
        torch.tensor(src_padded, dtype=torch.long, device=device),
        torch.tensor(tgt_input_padded, dtype=torch.long, device=device),
        torch.tensor(tgt_output_padded, dtype=torch.long, device=device),
        torch.tensor(src_mask, dtype=torch.bool, device=device)
    )


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '..')
    from device_utils import get_device
    
    device = get_device()
    
    model = AdditionTransformer(
        d_model=128,
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=3,
        d_ff=512
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test with a few samples
    samples = [create_addition_sample(5) for _ in range(4)]
    print("\nSample data:")
    for eq, ans in samples:
        print(f"  {eq} → {ans}")
    
    # Collate and forward
    src, tgt_in, tgt_out, src_mask = collate_batch(samples, device)
    print(f"\nBatch shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt_in: {tgt_in.shape}")
    print(f"  tgt_out: {tgt_out.shape}")
    
    # Forward pass
    logits = model(src, tgt_in, src_mask)
    print(f"  logits: {logits.shape}")
    
    # Test generation
    print("\nGeneration test:")
    model.eval()
    for eq, expected in samples[:2]:
        src_single = torch.tensor([model.tokenize(eq)], device=device)
        output = model.generate(src_single, max_len=10)
        generated = model.detokenize(output[0].tolist())
        print(f"  {eq} → {generated} (expected: {expected})")
