import torch
import torch.nn as nn
from torch.nn import functional as F

class DanteBaseline(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_window, ff_expansion_factor=4, dropout_rate=0.1):
		super().__init__()
		self.context_window = context_window
		self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
		self.pos_emb = nn.Embedding(context_window, embedding_dim)
		self.net = nn.Sequential(
			nn.Linear(embedding_dim, ff_expansion_factor * embedding_dim),
			nn.GELU(),
			nn.Linear(ff_expansion_factor * embedding_dim, embedding_dim),
			nn.Dropout(dropout_rate),
			nn.Linear(embedding_dim, vocab_size)
		)

	def forward(self, idx):
		# Truncate to context_window
		idx_truncated = idx[:, -self.context_window:] # (B, min(idx.shape(1), context_window))
		T = idx_truncated.shape[1]
		
		tok_emb = self.tok_emb(idx_truncated)  # (B, T, C)
		pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
		emb = tok_emb + pos_emb # (B, T, C) + (T, C) --> (B, T, C) : broadcasting by coping the range B times
		return self.net(emb)

	def compute_loss(self, idx, targets):
		logits = self(idx)
		B, T, C = logits.shape
		logits = logits.view(B*T, C)
		targets = targets.view(B*T)
		loss = F.cross_entropy(logits, targets)
		return loss

	def generate(self, context, max_new_tokens=32):
		for _ in range(max_new_tokens):
			# Constrain to context window
			tokens_cond = context[:, -self.context_window:]
			logits = self(tokens_cond)
			logits = logits[:, -1, :] # B, C
			probs = F.softmax(logits, dim=-1) # B, C
			idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			context = torch.cat((context, idx_next), dim=1) # idx: (B, T) --> (B, T+1)
		return context