import torch
import torch.nn as nn


class VAE_encoder(nn.Module):
    def __init__(self, input_dim, num_hidden=2):
        super().__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.layer_1 = nn.Linear(self.input_dim, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_3 = nn.Linear(64, self.num_hidden)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return x


class VAE_decoder(nn.Module):
    def __init__(self, output_dim, num_hidden=2):
        super().__init__()
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.layer_1 = nn.Linear(self.num_hidden, 64)
        self.layer_2 = nn.Linear(64, 256)
        self.layer_3 = nn.Linear(256, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return x


class VAE(nn.Module):
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encode = VAE_encoder(input_dim=784)
        self.decode = VAE_decoder(output_dim=784)
        self.mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.log_var = nn.Linear(self.hidden_dim, self.hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encode(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return encoded, decoded, mu, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, 2)
        return self.decode(z)
