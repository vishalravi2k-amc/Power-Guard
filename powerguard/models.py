"""Small CPU-friendly networks with explicit tensor shapes."""
import torch
from torch import nn


class LoadLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden+4, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, history, target_calendar):
        # history: [batch, 48 hours, load + four calendar features]
        output, _ = self.lstm(history)
        return self.head(torch.cat([output[:, -1], target_calendar], dim=1)).squeeze(1)


class LoadAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3), nn.ReLU(),
                                 nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, x):
        return self.net(x)
