from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class AbsMessageAggregator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMessageAggregator, self).__init__()

    @abstractmethod
    def aggregate(self, nodes: np.ndarray, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: np.ndarray
                shape = (batch_size,)
            messages: torch.Tensor
                shape = (batch_size, message_dim)

        Returns:
            aggr_messages: torch.Tensor
                shape = (n_unique_nodes, message_dim)
        """
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
