from typing import Any

import torch
import numpy as np


class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super(TimeEncoder, self).__init__()

        self._weight = torch.nn.Linear(1, dimension)

        self._weight.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1)
        )
        self._weight.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(dim=-1)  # (batch_size, seq_len) => (batch_size, seq_len, 1)
        output = torch.cos(self._weight(input))  # (batch_size, seq_len, 1) => (batch_size, seq_len, dimension)
        return output

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
