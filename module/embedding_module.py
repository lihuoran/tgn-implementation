from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import torch
from torch import nn

from data.data import AbsFeatureRepo
from utils import NeighborFinder


class AbsEmbeddingModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super(AbsEmbeddingModule, self).__init__()

        self._num_nodes = num_nodes
        self._embedding_dim = embedding_dim

        self._feature_repo: Optional[AbsFeatureRepo] = None
        self._neighbor_finder: Optional[NeighborFinder] = None

    @abstractmethod
    def compute_embedding(
        self,
        nodes: torch.Tensor,
        timestamps: torch.Tensor,
        memory: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def set_feature_repo(self, feature_repo: AbsFeatureRepo) -> None:
        self._feature_repo = feature_repo

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._neighbor_finder = neighbor_finder

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class SimpleEmbeddingModule(AbsEmbeddingModule):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super(SimpleEmbeddingModule, self).__init__(num_nodes, embedding_dim)

        self._embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)

    @abstractmethod
    def compute_embedding(
        self,
        nodes: torch.Tensor,
        timestamps: torch.Tensor,
        memory: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._embedding(nodes)
