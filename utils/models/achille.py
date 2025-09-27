from torch import nn
import torch.nn.functional as F

class Achille_MNIST_FC(nn.Module):
    def __init__(self, input_size=32):
        super(Achille_MNIST_FC, self).__init__()
        self.flatten = nn.Flatten()
        
        self.hidden1 = nn.Linear(input_size*input_size, 2500)
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
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.hidden1(x)))
        x = F.relu(self.bn2(self.hidden2(x)))
        x = F.relu(self.bn3(self.hidden3(x)))
        x = F.relu(self.bn4(self.hidden4(x)))
        x = F.relu(self.bn5(self.hidden5(x)))
        x = self.output(x)
        return x