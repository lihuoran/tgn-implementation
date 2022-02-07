from typing import Any

import torch


class MergeLayer(torch.nn.Module):
    def __init__(self, input_dim_1: int, input_dim_2: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self._fc1 = torch.nn.Linear(input_dim_1 + input_dim_2, hidden_dim)
        self._fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self._activation = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self._fc1.weight)
        torch.nn.init.xavier_normal_(self._fc2.weight)

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_1, input_2], dim=1)
        h = self._activation(self._fc1(x))
        return self._fc2(h)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
