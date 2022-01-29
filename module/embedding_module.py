from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from utils import NeighborFinder


class AbsEmbeddingModule(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsEmbeddingModule, self).__init__()
        self._neighbor_finder: Optional[NeighborFinder] = None

    @abstractmethod
    def compute_embedding(
        self,
        nodes: np.ndarray,
        timestamps: np.ndarray,
        n_layers: int,
        n_neighbors: int = 20,
        memory: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
        use_time_proj: bool = True,  # TODO: necessary?
    ) -> torch.Tensor:
        raise NotImplementedError

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._neighbor_finder = neighbor_finder

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
