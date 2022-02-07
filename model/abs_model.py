from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import torch
from torch import nn

from data.data import AbsFeatureRepo, DataBatch
from utils.training import NeighborFinder


class AbsModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, feature_repo: AbsFeatureRepo, device: torch.device) -> None:
        super(AbsModel, self).__init__()
        self._feature_repo = feature_repo
        self._device = device
        self._num_nodes = feature_repo.num_nodes()
        self._node_feature_dim = feature_repo.node_feature_dim()
        self._edge_feature_dim = feature_repo.edge_feature_dim()

    @abstractmethod
    def train_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval_mode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_temporal_embeddings(self, batch: DataBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_edge_probabilities(self, batch: DataBatch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
