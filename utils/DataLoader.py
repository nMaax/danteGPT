import torch

def get_batch(data, batch_size, block_size, device=None):

  ix = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,), device=device)
  x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)

  return x, y