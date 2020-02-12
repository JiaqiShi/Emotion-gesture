import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import time
import numpy as np

# from datasets import SimpleSet, SimpleEmoSet
# from callbacks import EarlyStopping


class AudioOnly(nn.Module):
    def __init__(self,
                 hidden_dim,
                 input_dim,
                 output_dim,
                 *,
                 num_stack=1,
                 return_sequences=False,
                 return_states=False):
        super(AudioOnly, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_stack)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def compile(self, *, lr, optimizer, criterion, device=None):
        if device:
            self.device = device
            self.to(device)
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        if criterion == 'mse':
            self.criterion = nn.MSELoss()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            if self.device:
                x_batch = x_batch.to(self.device)
                y_true_batch = y_true_batch.to(self.device)
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
            if self.device:
                x_batch = x_batch.to(self.device)
                y_true_batch = y_true_batch.to(self.device)
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

    def fit(self,
            train_set,
            valid_set,
            *,
            epochs=1,
            batch_size=1,
            shuffle=True,
            stopping=None):
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=shuffle)
        self.hist = {'train_loss': [], 'valid_loss': []}
        self.stopping = stopping
        self.stop = False
        print("Train on {} samples -- validate on {} samples".format(
            len(train_loader), len(valid_loader)))
        for epoch in range(1, epochs+1):
            self._on_epoch_begin()
            self._train_step(train_loader)
            self._valid_step(valid_loader)
            self._on_epoch_end(epoch)
            if self.stop:
                break
        return self.hist

    def fit_generator(self,
                      train_generator,
                      valid_generator,
                      *,
                      epochs=1,
                      stopping=None):
        self.hist = {'train_loss': [], 'valid_loss': []}
        self.stopping = stopping
        self.stop = False
        print("Train on {} batches -- validate on {} batches".format(
            len(train_generator), len(valid_generator)))
        for epoch in range(1, epochs+1):
            self._on_epoch_begin()
            self._train_step(train_generator)
            self._valid_step(valid_generator)
            self._on_epoch_end(epoch)
            if self.stop:
                break
        return self.hist


class AudioEmotion(nn.Module):
    def __init__(self,
                 hidden_dim,
                 x_input_dim,
                 e_input_dim,
                 output_dim,
                 *,
                 num_stack=1,
                 return_sequences=False,
                 return_states=False):
        super(AudioEmotion, self).__init__()
        self.x_input_dim = x_input_dim
        self.e_input_dim = e_input_dim
        self.input_dim = x_input_dim + e_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.return_states = return_states
        self.fc_mfc = nn.Linear(x_input_dim, hidden_dim)
        self.fc_emo = nn.Linear(e_input_dim, hidden_dim)
        self.gru = nn.GRU(2*hidden_dim, 2*hidden_dim, num_layers=num_stack)
        self.fc_out = nn.Linear(2*hidden_dim, output_dim)

    def compile(self, *, lr, optimizer, criterion, device=None):
        
        if device:
            self.device = device
            self.to(device)
        else:
            self.device = None
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        if criterion == 'mse':
            self.criterion = nn.MSELoss()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, e):
        assert len(
            x.size()) == 3, "x input must be with shape (batch, time_step, dim)"
        assert len(e.size()) == 2, "e input must be with shape (batch, num_cats)"
        
        e = e.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.relu(self.fc_mfc(x))
        e = torch.relu(self.fc_emo(e))
        x = torch.cat([x, e], -1)
        x, states = self.gru(x.view(x.size(1), x.size(0), x.size(2)))
        if self.return_sequences:
            x = torch.relu(x.view(x.size(1), x.size(0), x.size(2)))
        else:
            x = torch.relu(x[-1])
        x = self.fc_out(x)
        if self.return_states:
            return x, states
        else:
            return x

    def _on_epoch_begin(self):
        self.start_time = time.time()

    def _train_step(self, train_loader):
        self.train()
        losses = []
        for x_batch, e_batch, y_true_batch in train_loader:
            if self.device:
                x_batch = x_batch.to(self.device)
                e_batch = e_batch.to(self.device)
                y_true_batch = y_true_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred_batch = self.forward(x_batch, e_batch)
            loss = self.criterion(y_pred_batch, y_true_batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.train_loss = np.mean(losses)

    def _valid_step(self, valid_loader):
        self.eval()
        losses = []
        for x_batch, e_batch, y_true_batch in valid_loader:
            if self.device:
                x_batch = x_batch.to(self.device)
                e_batch = e_batch.to(self.device)
                y_true_batch = y_true_batch.to(self.device)
            y_pred_batch = self.forward(x_batch, e_batch)
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

    def fit(self,
            train_set,
            valid_set,
            *,
            epochs=1,
            batch_size=1,
            shuffle=True,
            stopping=None):
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=shuffle)
        self.hist = {'train_loss': [], 'valid_loss': []}
        self.stopping = stopping
        self.stop = False
        print("Train on {} samples -- validate on {} samples".format(
            len(train_loader), len(valid_loader)))
        for epoch in range(1, epochs+1):
            self._on_epoch_begin()
            self._train_step(train_loader)
            self._valid_step(valid_loader)
            self._on_epoch_end(epoch)
            if self.stop:
                break
        return self.hist

    def fit_generator(self,
                      train_generator,
                      valid_generator,
                      *,
                      epochs=1,
                      stopping=None):
        self.hist = {'train_loss': [], 'valid_loss': []}
        self.stopping = stopping
        self.stop = False
        print("Train on {} batches -- validate on {} batches".format(
            len(train_generator), len(valid_generator)))
        for epoch in range(1, epochs+1):
            self._on_epoch_begin()
            self._train_step(train_generator)
            self._valid_step(valid_generator)
            self._on_epoch_end(epoch)
            if self.stop:
                break
        return self.hist


if __name__ == "__main__":

    # net = AudioOnly(hidden_dim=128, input_dim=30, output_dim=16,
    #                 num_stack=2, return_sequences=True, return_states=False)
    # print(net)

    # net.compile(lr=1e-4, optimizer='adam', criterion='mse')

    # stopping = EarlyStopping(verbose=1, name='chkpt.pt')

    # X = np.zeros((1, 300, 30))
    # y = np.zeros((1, 300, 16))

    # train_set = SimpleSet(X, y)

    # net.fit(train_set, train_set, epochs=10, stopping=stopping)

    net = AudioEmotion(hidden_dim=128, x_input_dim=20, e_input_dim=11, output_dim=30,
                    num_stack=2, return_sequences=True, return_states=False)

    print(net)

    net.compile(lr=1e-4, optimizer='adam', criterion='mse')

    X = torch.zeros((10, 300, 20))
    e = torch.zeros((10, 12))
    y = torch.zeros((10, 300, 30))

    train_set = SimpleEmoSet(X, e, y)

    net.fit(train_set, train_set)

