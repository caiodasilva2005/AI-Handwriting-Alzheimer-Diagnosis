"""
MLP Architecture 
"""
import torch
import torch.nn as nn


class KinematicMLP(nn.Module):

    def __init__(self, input_size, dropout_rate=0.3):
        super(KinematicMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)