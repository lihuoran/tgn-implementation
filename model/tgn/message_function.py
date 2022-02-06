from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn


class AbsMessageFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMessageFunction, self).__init__()

        self._fc = nn.Linear(10, 5)

    @abstractmethod
    def compute_message(self, raw_message: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
