"""Small MLP regressor: pulse-wave features → SBP / DBP."""
import torch
import torch.nn as nn

FEATURE_DIM = 18


class BPRegressor(nn.Module):
    def __init__(self, n_features=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)
