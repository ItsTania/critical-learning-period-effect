import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def get_activation_module(name: str) -> nn.Module:
    """Return the activation as an nn.Module based on a string name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")

class CNN(nn.Module):
    def __init__(self, input_dim, input_channels=1, num_classes=10, activation="relu"):
        super(CNN, self).__init__()

        self.activation = get_activation_module(activation)

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate required size
        with torch.no_grad():
            input_size = int(math.sqrt(input_dim))
            assert input_size * input_size == input_dim, f"Input dimension {input_dim} is not a perfect square!"

            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            x = self.layer1(dummy_input)
            x = self.layer2(x)
            flattened_size = x.numel() // x.shape[0]

        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = self.fc(x)
        return x