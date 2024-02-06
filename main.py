import torch
from data.load_data import Data
from model.model import VAE

data = Data()
train_loader, test_loader = data.get_dataloaders()
data.show_images(train_loader)
model = VAE(hidden_dim=2)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            _, outputs, _, _ = model(inputs)
            loss = loss_fn(outputs, torch.nn.Flatten()(inputs))  # + kl_inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # Evaluate on validation set after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in enumerate(test_loader):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
        print("val_loss", val_loss / len(test_loader))


train(num_epochs=10)
