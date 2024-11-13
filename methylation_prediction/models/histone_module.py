# models/histone_module.py
import torch.nn as nn
import torch.nn.functional as F

class HistoneModule(nn.Module):
    def __init__(self, config):
        super(HistoneModule, self).__init__()
        self.fc1 = nn.Linear(6, config['histone_hidden_size'])
        self.activation = getattr(F, config['activation_func'])

    def forward(self, features):
        x = self.activation(self.fc1(features))
        return x
