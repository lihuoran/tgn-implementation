from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from model.tgn.memory import Message


class AbsMessageAggregator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMessageAggregator, self).__init__()

    @abstractmethod
    def aggregate(
        self, nodes: torch.Tensor,
        messages: Dict[int, List[Message]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
