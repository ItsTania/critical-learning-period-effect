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
class Achille_MNIST_FC(nn.Module):
    def __init__(self, input_dim, activation="relu", input_channels=None):
        super(Achille_MNIST_FC, self).__init__()
        self.flatten = nn.Flatten()
        
        self.hidden1 = nn.Linear(input_dim, 2500)
        self.bn1 = nn.BatchNorm1d(2500)
        
        self.hidden2 = nn.Linear(2500, 2000)
        self.bn2 = nn.BatchNorm1d(2000)
        
        self.hidden3 = nn.Linear(2000, 1500)
        self.bn3 = nn.BatchNorm1d(1500)
        
        self.hidden4 = nn.Linear(1500, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        
        self.hidden5 = nn.Linear(1000, 500)
        self.bn5 = nn.BatchNorm1d(500)
        
        self.output = nn.Linear(500, 10) 
        self.activation = get_activation(activation)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.bn1(self.hidden1(x)))
        x = self.activation(self.bn2(self.hidden2(x)))
        x = self.activation(self.bn3(self.hidden3(x)))
        x = self.activation(self.bn4(self.hidden4(x)))
        x = self.activation(self.bn5(self.hidden5(x)))
        x = self.output(x)
        return x

class Achille_MNIST_FC_No_BatchNorm(nn.Module):
    def __init__(self, input_dim, activation="relu", input_channels=None):
        super(Achille_MNIST_FC_No_BatchNorm, self).__init__()
        self.flatten = nn.Flatten()
        
        self.hidden1 = nn.Linear(input_dim, 2500)
        self.hidden2 = nn.Linear(2500, 2000)
        self.hidden3 = nn.Linear(2000, 1500)
        self.hidden4 = nn.Linear(1500, 1000)
        self.hidden5 = nn.Linear(1000, 500)
        self.output = nn.Linear(500, 10)  
        
        self.activation = get_activation(activation)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.activation(self.hidden4(x))
        x = self.activation(self.hidden5(x))
        x = self.output(x)
        return x
