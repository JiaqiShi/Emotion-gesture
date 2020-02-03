import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from helpers import EarlyStopping, MyDataset
import time
import numpy as np


class Net(nn.Module):
    def __init__(self,
                 *,
                 hidden_dim,
                 input_dim,
                 output_dim,
                 num_stack=1,
                 return_sequences=False,
                 return_states=False
                 ):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_stack)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def compile(self, *, lr, optimizer, criterion):
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        if criterion == 'mse':
            self.criterion = nn.MSELoss()

    def forward(self, x):
        assert len(
            x.size()) == 3, "Input must be with shape (batch, time_step, dim)"

        x, states = self.gru(x.view(x.size(1), x.size(0), x.size(2)))
        if self.return_sequences:
            x = x.view(x.size(1), x.size(0), x.size(2))
        else:
            x = x[-1]
        x = self.fc(x)
        if self.return_states:
            return x, states
        else:
            return x

    def _on_epoch_begin(self):
        self.start_time = time.time()

    def _train_step(self, train_loader):
        self.train()
        losses = []
        for x_batch, y_true_batch in train_loader:
            self.optimizer.zero_grad()
            y_pred_batch = self.forward(x_batch)
            loss = self.criterion(y_pred_batch, y_true_batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.train_loss = np.mean(losses)

    def _valid_step(self, valid_loader):
        self.eval()
        losses = []
        for x_batch, y_true_batch in valid_loader:
            y_pred_batch = self.forward(x_batch)
            loss = self.criterion(y_pred_batch, y_true_batch)
            losses.append(loss.item())
        self.valid_loss = np.mean(losses)

    def _on_epoch_end(self, epoch):
        self.hist['train_loss'].append(self.train_loss)
        self.hist['valid_loss'].append(self.valid_loss)

        template = "Epoch {0} {1}s: loss: {2:.5f} - val_loss: {3:.5f}"
        print(template.format(
            epoch,
            int(time.time() - self.start_time),
            self.train_loss,
            self.valid_loss))

        if self.stopping:
            self.stopping(self.valid_loss, self)
            if self.stopping.early_stop:
                print("Earlystopping threshold")
                self.stop = True

    def fit(self, train_set, valid_set, *, epochs=1, batch_size=1, shuffle=True, stopping=None):

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=shuffle)

        self.hist = {'train_loss': [], 'valid_loss': []}
        self.stopping = stopping
        self.stop = False

        print("Train on {} samples -- validate on {} samples".format(len(train_loader), len(valid_loader)))

        for epoch in range(1, epochs+1):
            self._on_epoch_begin()
            self._train_step(train_loader)
            self._valid_step(valid_loader)
            self._on_epoch_end(epoch)
            if self.stop:
                break

        return self.hist


if __name__ == "__main__":

    net = Net(hidden_dim=128, input_dim=30, output_dim=16,
              num_stack=2, return_sequences=True, return_states=False)
    print(net)

    net.compile(lr=1e-4, optimizer='adam', criterion='mse')

    stopping = EarlyStopping(verbose=1)

    X = np.zeros((1, 300, 30))
    y = np.zeros((1, 300, 16))

    train_set = MyDataset(X, y)

    net.fit(train_set, train_set, epochs=10, stopping=stopping)
