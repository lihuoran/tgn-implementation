from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from module.memory import Message


class AbsMessageAggregator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMessageAggregator, self).__init__()

    @abstractmethod
    def aggregate(
        self, nodes: np.ndarray,
        messages: Dict[int, List[Message]]
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes: np.ndarray
                shape = (batch_size,)
            messages: Dict[int, List[Message]]

        Returns:
            unique_nodes: np.ndarray
            unique_message: torch.Tensor
            unique_ts: torch.Tensor
        """
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
