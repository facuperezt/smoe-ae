import torch
import torch.nn as nn
import torch.optim as optim
from models import Asereje
from data import DataLoader


# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define your dataloader
train_loader = DataLoader()
train_loader.initialize()

# Define your model
model = Asereje("models/config_files/base_config.json", device=device)
model: nn.Module

# Define your loss function
criterion = model.loss

# Define your optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 100
historic_loss = []
# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    
    # Iterate over the training dataset
    for i, (inputs, labels) in enumerate(train_loader.get(data="train")):
        print(i)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
    
    # Print the loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save the model if the loss is lower than the historic loss
    if loss < historic_loss[-1]:
        # Save the trained model
        torch.save(model.state_dict(), f'models/saves/last_run/_epoch_{epoch}.pth')
    
    historic_loss.append(loss.item())
    if all([abs(new_loss) >= abs(old_loss) for new_loss, old_loss in zip(historic_loss[-5:], historic_loss[-6:-1])]):
        break

torch.save(model.state_dict(), "models/saves/last_run/trained_model.pth")
