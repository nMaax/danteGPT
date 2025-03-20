import torch
import torch.nn as nn
from torch.nn import functional as F

class naiveLM(nn.Module):
  def __init__(self, vocab_size, latent_space_dim):
    super().__init__()
    self.context_window_size = None
    self.embedding_table = nn.Embedding(vocab_size, latent_space_dim)
    self.net = nn.Sequential(
        nn.Linear(in_features=latent_space_dim, out_features=latent_space_dim, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=latent_space_dim, out_features=latent_space_dim, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=latent_space_dim, out_features=latent_space_dim, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=latent_space_dim, out_features=vocab_size, bias=True)
    )

  def forward(self, idx):
    emb = self.embedding_table(idx)
    logits = self.net(emb)
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
      logits = self.forward(tokens) # B, T, C
      logits = logits[:, -1, :] # B, C
      probs = F.softmax(logits, dim=-1) # B, C
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      tokens = torch.cat((tokens, idx_next), dim=1) # idx: (B, T) --> (B, T+1)
    return tokens