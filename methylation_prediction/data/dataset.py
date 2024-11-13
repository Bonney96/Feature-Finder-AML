# data/dataset.py
import torch
from torch.utils.data import Dataset

class MethylationDataset(Dataset):
    def __init__(self, sequences, features, labels):
        self.sequences = sequences
        self.features = features
        self.labels = labels
        self.sequence_lengths = (sequences != 4).sum(axis=1)  # Assuming padding token is 4

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features = self.features[idx]
        label = self.labels[idx]
        seq_len = self.sequence_lengths[idx]
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }
