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

    @abstractmethod
    def message_dim(self) -> int:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class MLPMessageFunction(AbsMessageFunction):
    def __init__(self, raw_message_dim: int, message_dim: int) -> None:
        super(MLPMessageFunction, self).__init__()

        self._message_dim = message_dim
        self._fc = nn.Sequential(
            nn.Linear(raw_message_dim, raw_message_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dim // 2, message_dim)
        )

    def compute_message(self, raw_message: torch.Tensor) -> torch.Tensor:
        return self._fc(raw_message)

    def message_dim(self) -> int:
        return self._message_dim


class IdentityMessageFunction(AbsMessageFunction):
    def __init__(self, raw_message_dim: int) -> None:
        super(IdentityMessageFunction, self).__init__()
        self._raw_message_dim = raw_message_dim

    def compute_message(self, raw_message: torch.Tensor) -> torch.Tensor:
        return raw_message

    def message_dim(self) -> int:
        return self._raw_message_dim
