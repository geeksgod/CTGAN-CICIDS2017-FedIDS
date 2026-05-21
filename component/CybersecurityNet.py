import torch.nn as nn
from typing import List

class CybersecurityNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(h)]
            prev = h
        layers += [nn.Linear(prev, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)