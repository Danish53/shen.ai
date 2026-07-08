"""1D CNN on pulse waveform — Shen.ai-style deep learning BP path."""
import torch
import torch.nn as nn

WAVE_LEN = 256


class BPWaveNet(nn.Module):
    """Conv1D encoder on normalized PPG → SBP/DBP residuals from population baseline."""

    def __init__(self, wave_len=WAVE_LEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 2),
        )
        self.wave_len = wave_len

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x).squeeze(-1)
        return self.head(h)


def normalize_waveform(wave, target_len=WAVE_LEN):
    """Resample + z-score for CNN input."""
    y = torch.as_tensor(wave, dtype=torch.float32).flatten()
    if y.numel() < 20:
        return None
    if y.numel() != target_len:
        x_old = torch.linspace(0, 1, y.numel())
        x_new = torch.linspace(0, 1, target_len)
        y = torch.nn.functional.interpolate(
            y.reshape(1, 1, -1), size=target_len, mode="linear", align_corners=True,
        ).reshape(-1)
    y = y - y.mean()
    std = y.std()
    if std > 1e-6:
        y = y / std
    return y.reshape(1, -1)
