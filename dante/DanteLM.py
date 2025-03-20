import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, tokens_in, head_size, context_window):
    super().__init__()
    self.feat_in = tokens_in
    self.head_size = head_size
    self.context_window = context_window
    self.key = nn.Linear(tokens_in, head_size, bias=False)
    self.query = nn.Linear(tokens_in, head_size, bias=False)
    self.value = nn.Linear(tokens_in, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

  def forward(self, X):

    B, T, C = X.shape

    K = self.key(X)
    Q = self.query(X)
    V = self.value(X)

    self_attention = K @ Q.transpose(-1, -2) * self.head_size**-0.5
    self_attention = self_attention.masked_fill(torch.tril(torch.ones(T, T, device=X.device) == 0), float('-inf'))
    self_attention = F.softmax(self_attention, dim=-1)

    return self_attention @ V

class MultiHead(nn.Module):
  def __init__(self, num_heads, latent_space_dim, context_window):
    super().__init__()
    head_size = latent_space_dim // num_heads
    self.num_heads = num_heads
    self.context_window = context_window
    self.heads = nn.ModuleList([
        Head(tokens_in=latent_space_dim, head_size=head_size, context_window=context_window) for _ in range(num_heads)
        ])
    self.proj = nn.Linear(latent_space_dim, latent_space_dim)

  def forward(self, X):
    out = torch.cat([h(X) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out
  
class FeedForward(nn.Module):
  def __init__(self, latent_space_dim, hidden_layer_expansion_factor=4):
    super().__init__()
    self.latent_space_dim = latent_space_dim
    self.hidden_layer_expansion_factor = hidden_layer_expansion_factor
    self.net = nn.Sequential(
            nn.Linear(latent_space_dim, hidden_layer_expansion_factor*latent_space_dim),
            nn.GELU(),
            nn.Linear(hidden_layer_expansion_factor*latent_space_dim, latent_space_dim),
        )

  def forward(self, X):
    return self.net(X)
  

class Block(nn.Module):
  def __init__(self, latent_space_dim, num_heads, context_window, dropout_rate):
    super().__init__()
    self.self_attention = MultiHead(num_heads=num_heads, latent_space_dim=latent_space_dim, context_window=context_window)
    self.ff = FeedForward(latent_space_dim=latent_space_dim)

    self.ln1 = nn.LayerNorm(latent_space_dim) # LayerNorm before Attention
    self.ln2 = nn.LayerNorm(latent_space_dim) # LayerNorm before FF
    
    self.do1 = nn.Dropout(p=dropout_rate)  # Dropout after Attention
    self.do2 = nn.Dropout(p=dropout_rate)  # Dropout after FF


  def forward(self, x):
    x = x + self.do1(self.self_attention(self.ln1(x)))
    x = x + self.do2(self.ff(self.ln2(x)))
    return x
  
class DanteLM(nn.Module):
  def __init__(self, vocab_size, context_window, latent_space_dim, num_heads, n_blocks, dropout_rate=0.1):
    super().__init__()

    self.vocab_size = vocab_size
    self.context_window_size = context_window
    self.latent_space_dim = latent_space_dim
    self.num_heads = num_heads
    self.n_blocks = n_blocks

    self.tok_embedding_table = nn.Embedding(vocab_size, latent_space_dim)
    self.pos_embedding_table = nn.Embedding(context_window, latent_space_dim)
    self.blocks = nn.Sequential(
        *[Block(latent_space_dim=latent_space_dim, num_heads=num_heads, context_window=context_window, dropout_rate=dropout_rate) for _ in range(n_blocks)]
    )
    self.ln_f = nn.LayerNorm(latent_space_dim)
    self.lm_head = nn.Linear(latent_space_dim, vocab_size)

  def forward(self, idx):
    B, T = idx.shape

    tok_emb = self.tok_embedding_table(idx) # (B, T, C)
    pos_emb = self.pos_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
    emb = tok_emb + pos_emb # (B, T, C)
    x = self.blocks(emb)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)
    return logits

  def compute_loss(self, idx, targets):
    logits = self.forward(idx)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

  def generate(self, tokens, max_new_tokens=32):
    for _ in range(max_new_tokens):
      logits = self.forward(tokens[:, -self.context_window_size:]) # B, T, C
      logits = logits[:, -1, :] # B, C
      probs = F.softmax(logits, dim=-1) # B, C
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      tokens = torch.cat((tokens, idx_next), dim=1) # idx: (B, T) --> (B, T+1)
    return tokens