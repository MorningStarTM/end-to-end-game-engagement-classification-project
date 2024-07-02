import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class NeuralNetwork(nn.Module):
    """
    Model class 
    """
    def __init__(self, input_dim, out_dim, hidden_dim):

        """
        Args:
            input dim (int) : input data dimension
            out_dim (int) : output data dimension
            hidden_dim (int) : hidden dimension
        
       """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    

        
