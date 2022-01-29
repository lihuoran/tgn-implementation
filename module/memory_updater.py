from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from torch import nn

from module.memory import Memory


class AbsMemoryUpdater(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMemoryUpdater, self).__init__()

    @abstractmethod
    def update_memory(
        self, memory: Memory, unique_nodes: np.ndarray, unique_messages: torch.Tensor, unique_ts: torch.Tensor
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_updated_memory(
        self, memory: Memory, unique_nodes: np.ndarray, unique_messages: torch.Tensor, unique_ts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
