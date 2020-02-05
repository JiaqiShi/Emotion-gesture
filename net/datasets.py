import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SimpleSet(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DynamicPaddingSet(Dataset):
    def __init__(self, X, y, batch_size):
        self.X = list(map(lambda x: torch.tensor(x, dtype=torch.float), X))
        self.y = list(map(lambda x: torch.tensor(x, dtype=torch.float), y))
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        # batch idx is [idx:idx+self.batch_size]
        Xs = self.X[idx:idx+self.batch_size]
        ys = self.y[idx:idx+self.batch_size]
        # padding
        Xs = pad_sequence(Xs, batch_first=True)
        ys = pad_sequence(ys, batch_first=True)
        return Xs, ys
