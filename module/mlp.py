from typing import Any

import torch


class MLP(torch.nn.Module):
    def __init__(self, dim: int, drop: float = 0.3) -> None:
        super().__init__()
        self._fc1 = torch.nn.Linear(dim, 80)
        self._fc2 = torch.nn.Linear(80, 10)
        self._fc3 = torch.nn.Linear(10, 1)
        self._activation = torch.nn.ReLU()
        self._dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dropout(self._activation(self._fc1(x)))
        x = self._dropout(self._activation(self._fc2(x)))
        return self._fc3(x).squeeze(dim=1)  # (batch_size,)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
