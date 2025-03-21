import torch
import matplotlib.pyplot as plt
from utils.DataLoader import get_batch

def train_model(model, train_data, test_data, optimizer, epochs, eval_every=1000, batch_size=32, device=None):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model.to(device)  # Move model to the specified device

  train_loss_values = []
  test_loss_values = []

  for i in range(epochs):
    # Training
    xb, yb = get_batch(data=train_data, device=device, batch_size=batch_size)
    loss = model.compute_loss(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    train_loss_values.append(loss.item())

    # Evaluation
    if i % eval_every == 0:
      with torch.no_grad():
        xb, yb = get_batch(data=test_data, device=device)
        test_loss = model.compute_loss(xb, yb)
        test_loss_values.append(test_loss.item())
        print(f"Epoch {i}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

  return train_loss_values, test_loss_values

def plot_loss_functions(train_loss_values, test_loss_values, epochs, eval_every=1000):

  plt.figure(figsize=(10, 5))
  plt.plot(train_loss_values, label='Train Loss')
  plt.plot(range(0, epochs, eval_every), test_loss_values, label='Test Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Test Loss Curves')
  plt.legend()
  plt.show()