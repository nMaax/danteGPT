import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head  # Dimensionality of each attention head's output
        self.key = nn.Linear(d_model, d_head, bias=False)  # Key projection
        self.query = nn.Linear(d_model, d_head, bias=False)  # Query projection
        self.value = nn.Linear(d_model, d_head, bias=False)  # Value projection

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)    # (B, T, d_head): Project input to key space
        q = self.query(x)  # (B, T, d_head): Project input to query space
        
        # Compute attention scores (scaled dot-product attention)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        
        # Causal autoregressive mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        v = self.value(x)  # (B, T, d_head): Project input to value space
        return attn_weights @ v  # Weighted sum of values

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads  # Dimensionality of each head
        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_head) for _ in range(num_heads)])  # List of attention heads
        self.proj = nn.Linear(d_model, d_model)  # Linear projection to combine head outputs

    def forward(self, x):
        # Concatenate outputs of all heads and project to d_model
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        # Two-layer feedforward network with GELU activation
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),  # Expand dimensionality
            nn.GELU(),  # GELU activation
            nn.Linear(expansion_factor * d_model, d_model),  # Project back to d_model
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)  # Multi-head attention
        self.ff = FeedForward(d_model)  # Feedforward network
        self.ln1 = nn.LayerNorm(d_model)  # Layer norm before attention
        self.ln2 = nn.LayerNorm(d_model)  # Layer norm before feedforward
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after attention
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after feedforward

    def forward(self, x):
        # Residual connection around multi-head attention
        x = x + self.dropout1(self.attn(self.ln1(x)))
        # Residual connection around feedforward network
        x = x + self.dropout2(self.ff(self.ln2(x)))
        return x

class DanteTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_transformer_blocks, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model # Embedding space dimension
        self.block_size = block_size  # Maximum sequence length (block size)
        self.token_emb = nn.Embedding(vocab_size, d_model)  # Token embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)  # Positional embeddings
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, dropout_rate) 
            for _ in range(num_transformer_blocks)
        ])
        self.ln_final = nn.LayerNorm(d_model)  # Final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)  # Language model head

    def forward(self, idx):
        
        # Enforce block_size constraint
        idx = idx[:, -self.block_size:]
        B, T = idx.shape
        
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)
    
    def compute_loss(self, idx, targets):
        logits = self(idx)  # Compute logits
        # print(logits.shape, targets.shape)
        # print(logits.view(-1, logits.shape[-1]).shape, targets.view(-1).shape)
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))  # Compute cross-entropy loss

    def generate(self, context, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context to block_size
            context = context[:, -self.block_size:]
            logits = self(context)  # Compute logits
            next_logits = logits[:, -1, :]  # Focus on last token's logits
            probs = F.softmax(next_logits, dim=-1)  # Compute probabilities
            context = torch.cat([context, torch.multinomial(probs, num_samples=1)], dim=-1)  # Sample next token
        return context