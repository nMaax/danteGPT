import torch

def get_batch(data, batch_size=1024, block_size=256, device=None):
  # Use CUDA if available, otherwise CPU
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #data = train_data if split == 'train' else test_data
  ix = torch.randint(low=0, high=(len(data) - block_size), size=(batch_size,), device=device)
  x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
  #y = torch.clamp(y, 0, vocab_size - 1)
  return x, y