# Logistic Regression
import torch
import torch.nn as nn

class LogisticRegressionModule(nn.Module):
    """
    Simple logistic regression model for MNIST-style input.
    """

    def __init__(self, input_dim: int, output_dim: int = 10, activation=None, input_channels=None): # Always uses softmax
        """
        Args:
            input_dim: Flattened input size (
            output_dim: Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim) or (batch_size, C, H, W)
        """
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        logits = torch.sigmoid(self.linear(x))
        return logits