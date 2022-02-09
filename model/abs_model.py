from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import nn

from data import AbsFeatureRepo, DataBatch
from utils import NeighborFinder

EmbeddingBundle = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]


class AbsEmbeddingModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, feature_repo: AbsFeatureRepo, device: torch.device) -> None:
        super(AbsEmbeddingModel, self).__init__()
        self._feature_repo = feature_repo
        self._device = device
        self._num_nodes = feature_repo.num_nodes()
        self._node_feature_dim = feature_repo.node_feature_dim()
        self._edge_feature_dim = feature_repo.edge_feature_dim()

    def epoch_start_step(self) -> None:
        pass

    def epoch_end_step(self) -> None:
        pass

    def backward_post_step(self) -> None:
        pass

    @abstractmethod
    def compute_temporal_embeddings(self, batch: DataBatch) -> EmbeddingBundle:
        raise NotImplementedError

    @abstractmethod
    def compute_edge_probabilities(self, batch: DataBatch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class AbsBinaryClassificationModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, input_dim: int, device: torch.device) -> None:
        super(AbsBinaryClassificationModel, self).__init__()
        self._device = device
        self._input_dim = input_dim

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @abstractmethod
    def get_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
