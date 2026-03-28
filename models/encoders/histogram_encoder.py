import torch
from torch import nn


class HistogramEncoder(nn.Module):
    """Placeholder encoder for future histogram->symbolic decoding pipeline."""

    def __init__(self, input_bins: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_bins, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, hist: torch.Tensor):
        enc = self.net(hist)
        # Keep interface compatible with seq2seq encoder output shape.
        return enc.unsqueeze(1), None
