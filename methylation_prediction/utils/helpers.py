# utils/helpers.py
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def get_middle_sequence(sequence, maxlen):
    seq_len = len(sequence)
    if seq_len <= maxlen:
        return sequence
    else:
        start = (seq_len - maxlen) // 2
        return sequence[start:start + maxlen]

def pad_sequence(sequence, maxlen):
    seq_len = len(sequence)
    if seq_len >= maxlen:
        return sequence
    else:
        padding_needed = maxlen - seq_len
        padded_sequence = sequence + ('N' * padding_needed)
        return padded_sequence

def encode_sequences(sequences, maxlen):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded_seqs = []

    for idx, seq in enumerate(sequences):
        seq = seq.upper()

        # Handle long and short sequences
        if len(seq) > maxlen:
            seq = get_middle_sequence(seq, maxlen)
        else:
            seq = pad_sequence(seq, maxlen)

        seq_encoded = [mapping.get(nuc, 4) for nuc in seq]
        encoded_seqs.append(seq_encoded)

    return np.array(encoded_seqs)
