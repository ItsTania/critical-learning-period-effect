import torch
from torch import nn
import torch.nn.functional as F

def get_activation(name):
    """Return the activation function based on a string name."""
    name = name.lower()
    if name == "relu":
        return F.relu
    elif name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == "elu":
        return F.elu
    elif name == "gelu":
        return F.gelu
    elif name == "tanh":
        return torch.tanh
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
class BasicClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=784,
            hidden_dim=100,
            output_dim=10,
            number_of_layers=3,
            activation = 'relu',
            input_channels=None,
    ):
        super(BasicClassifierModule, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(number_of_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.activation = get_activation(activation)

    def forward(self, X, **kwargs):
        X = X.view(X.size(0), -1)
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        X = self.output(X)
        return X
    
class ChokepointClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=784,
            output_dim=10,
            number_of_layers=3,
            activation = 'relu',
            scale=4,
            input_channels=None
    ):
        super(ChokepointClassifierModule, self).__init__()

        current_dim_size = int(input_dim/scale)
        layers = [nn.Linear(input_dim, current_dim_size)]
        for _ in range(number_of_layers - 1):
            layers.append(nn.Linear(current_dim_size, int(current_dim_size/scale)))
            current_dim_size = int(current_dim_size/scale)
        self.hidden_layers = nn.ModuleList(layers)
        self.output = nn.Linear(current_dim_size, output_dim)

        self.activation = get_activation(activation)

    def forward(self, X, **kwargs):
        X = X.view(X.size(0), -1)
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        X = self.output(X)
        return X
    
class BottleneckClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=784,
            output_dim=10,
            bottleneck=3,
            activation = F.relu,
            scale=4,
            input_channels=None
    ):
        super(BottleneckClassifierModule, self).__init__()

        current_dim = int(input_dim/scale)
        layers = [nn.Linear(input_dim, current_dim)]

        layers.append(nn.Linear(current_dim, int(current_dim/scale)))

        layers.append(nn.Linear(int(current_dim/scale), bottleneck)) # bottleneck

        layers.append(nn.Linear(bottleneck, int(current_dim/scale)))

        layers.append(nn.Linear(int(current_dim/scale), current_dim))

        self.hidden_layers = nn.ModuleList(layers)
        self.output = nn.Linear(current_dim, output_dim)

        self.activation = get_activation(activation)

    def forward(self, X, **kwargs):
        X = X.view(X.size(0), -1)
        for layer in self.hidden_layers:
            X = self.activation(layer(X))
        X = self.output(X)
        return X