# models/joint_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointModule(nn.Module):
    def __init__(self, dna_module, histone_module, config):
        super(JointModule, self).__init__()
        self.dna_module = dna_module
        self.histone_module = histone_module
        combined_size = dna_module.fc.out_features + histone_module.fc1.out_features
        self.fc1 = nn.Linear(combined_size, config['joint_hidden_size'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.fc2 = nn.Linear(config['joint_hidden_size'], 1)
        self.activation = getattr(F, config['activation_func'])

    def forward(self, sequence, sequence_lengths, features):
        dna_output = self.dna_module(sequence, sequence_lengths)
        histone_output = self.histone_module(features)
        x = torch.cat((dna_output, histone_output), dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)
