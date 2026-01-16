import matplotlib.pyplot as plt
import torch

from utils.DataLoader import get_batch


def train_model(model, train_data, test_data, optimizer, scheduler, epochs, batch_size, block_size, device, eval_every=1000):

  train_loss_values = []
  test_loss_values = []

  for i in range(epochs):
    # Training
    xb, yb = get_batch(data=train_data, batch_size=batch_size, block_size=block_size, device=device)
    loss = model.compute_loss(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    train_loss_values.append(loss.item())

    # Evaluation
    if test_data is not None and (i % eval_every == 0):
      with torch.no_grad():
        xb, yb = get_batch(data=test_data, batch_size=batch_size, block_size=block_size, device=device)
        test_loss = model.compute_loss(xb, yb)
        test_loss_values.append(test_loss.item())
        print(f"Epoch {i}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")
    else:
      if i % eval_every == 0:
        print(f"Epoch {i}: Train Loss = {loss.item():.4f}")
    
    if scheduler is not None:
      scheduler.step()

  return train_loss_values, test_loss_values if test_data is not None else None

def plot_loss_functions(train_loss_values, test_loss_values, epochs, eval_every=1000):

  plt.figure(figsize=(10, 5))
  plt.plot(train_loss_values, label='Train Loss')
  if test_loss_values is not None:
    plt.plot(range(0, epochs, eval_every), test_loss_values, label='Test Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Loss Curves')
  plt.legend()
  plt.show()