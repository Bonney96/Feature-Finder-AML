# models/dna_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DNAModule(nn.Module):
    def __init__(self, config):
        super(DNAModule, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=5,  # A, C, G, T, N
            embedding_dim=config['embedding_dim'],
            padding_idx=4
        )
        nn.init.normal_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[4].fill_(0)  # Mask padding nucleotides 'N'

        self.conv1 = nn.Conv1d(
            in_channels=config['embedding_dim'],
            out_channels=config['num_filters_conv1'],
            kernel_size=config['kernel_size_conv1'],
            padding=config['kernel_size_conv1'] // 2,
            bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=config['num_filters_conv1'],
            out_channels=config['num_filters_conv2'],
            kernel_size=config['kernel_size_conv2'],
            padding=config['kernel_size_conv2'] // 2,
            bias=False
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.activation = getattr(F, config['activation_func'])
        self.fc = nn.Linear(config['num_filters_conv2'], config['fc_size_dna'])

    def forward(self, sequence, sequence_lengths):
        x = self.embedding(sequence)
        x = x.permute(0, 2, 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc(x))
        return x