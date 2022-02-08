# File for Neural Network Model

import torch.nn as nn

# Example layout -- Code from Pytorch tutorial
class NN_Model(nn.Module):

    def __init__(self):
        super(NN_Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
        
    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)
        return logits