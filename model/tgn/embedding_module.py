from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.functional import embedding

from data.data import AbsFeatureRepo
from model.tgn.temporal_attention import TemporalAttentionLayer
from module.time_encoder import TimeEncoder
from utils.training import NeighborFinder


class AbsEmbeddingModule(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        embedding_dim: int,
    ) -> None:
        super(AbsEmbeddingModule, self).__init__()

        self._feature_repo = feature_repo
        self._device = device
        self._embedding_dim = embedding_dim

    @abstractmethod
    def compute_embedding(
        self,
        nodes: torch.Tensor,
        timestamps: torch.Tensor,
        memory_tensor: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._neighbor_finder = neighbor_finder


class SimpleEmbeddingModule(AbsEmbeddingModule):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        embedding_dim: int,
    ) -> None:
        super(SimpleEmbeddingModule, self).__init__(feature_repo, device, embedding_dim)

        self._embedding = nn.Parameter(torch.rand(feature_repo.num_nodes(), embedding_dim))

    def compute_embedding(
        self,
        nodes: torch.Tensor,
        timestamps: torch.Tensor,
        memory_tensor: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
    ) -> torch.Tensor:
        return embedding(nodes, self._embedding)


class GraphEmbedding(AbsEmbeddingModule):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        embedding_dim: int,
        n_layers: int,
        n_neighbors: int,
    ) -> None:
        super(GraphEmbedding, self).__init__(feature_repo, device, embedding_dim)

        self._n_layers = n_layers
        self._n_neighbors = n_neighbors
        self._time_encoder = TimeEncoder(feature_repo.node_feature_dim())

    def compute_embedding(
        self,
        nodes: torch.Tensor,
        timestamps: torch.Tensor,
        memory_tensor: torch.Tensor = None,
        time_diffs: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._compute_embedding_impl(
            self._n_layers, nodes, timestamps, memory_tensor=memory_tensor,
        )

    def _compute_embedding_impl(
        self,
        layer_id: int,
        nodes: torch.Tensor,  # (B,)
        timestamps: torch.Tensor,  # (B,)
        memory_tensor: torch.Tensor = None,  # (B, node_feat_dim)
    ) -> torch.Tensor:  # (B, node_feat_dim)
        node_time_emb = self._time_encoder(torch.zeros_like(timestamps).unsqueeze(1))  # (B, 1, node_feat_dim)
        node_features = torch.from_numpy(  # (B, node_feat_dim)
            self._feature_repo.get_node_feature(nodes.detach().numpy())
        ).float().to(self._device)
        if memory_tensor is not None:
            node_features = memory_tensor[nodes] + node_features  # (B, node_feat_dim)

        if layer_id == 0:
            return node_features
        else:
            neighbors = self._neighbor_finder.get_temporal_neighbor(
                nodes=nodes.detach().numpy(), timestamps=timestamps.detach().numpy(), k=self._n_neighbors
            )
            ts_numpy = np.array([[elem[0] for elem in neighbor_list] for neighbor_list in neighbors])
            edge_id_numpy = np.array([[elem[1] for elem in neighbor_list] for neighbor_list in neighbors])
            node_id_numpy = np.array([[elem[2] for elem in neighbor_list] for neighbor_list in neighbors])
            ts_torch = torch.from_numpy(ts_numpy).float().to(self._device)  # (B, n_neighbor)
            node_id_torch = torch.from_numpy(node_id_numpy).long().to(self._device)  # (B, n_neighbor)
            edge_deltas_torch = timestamps.unsqueeze(1) - ts_torch  # (B, n_neighbor)

            neighbor_emb = self._compute_embedding_impl(
                layer_id=layer_id - 1,
                nodes=node_id_torch.flatten(),  # (B * n_neighbor,)
                timestamps=timestamps.repeat_interleave(self._n_neighbors),  # (B * n_neighbor,)
                memory_tensor=memory_tensor,  # (B, node_feat_dim)
            )  # (B, node_feat_dim)

            effective_n_neighbors = max(self._n_neighbors, 1)
            neighbor_emb = neighbor_emb.view(len(nodes), effective_n_neighbors, -1)  # (B, node_feat_dim, 1)
            edge_time_emb = self._time_encoder(edge_deltas_torch)  # (B, node_feat_dim)
            edge_features = torch.from_numpy(  # (B, edge_feat_dim)
                self._feature_repo.get_edge_feature(edge_id_numpy)
            ).float().to(self._device)
            mask = (node_id_torch == 0)

            return self._aggregate(
                layer_id, node_features, node_time_emb, edge_time_emb, neighbor_emb, edge_features, mask
            )

    @abstractmethod
    def _aggregate(
        self,
        layer_id: int,
        node_features: torch.Tensor,
        node_time_emb: torch.Tensor,
        edge_time_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        embedding_dim: int,
        n_layers: int,
        n_neighbors: int,
        n_head: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(GraphAttentionEmbedding, self).__init__(
            feature_repo, device, embedding_dim, n_layers, n_neighbors
        )

        self._attention_models = torch.nn.ModuleList([
            TemporalAttentionLayer(
                node_feat_dim=feature_repo.node_feature_dim(),
                edge_feat_dim=feature_repo.edge_feature_dim(),
                neighbor_emb_dim=self.embedding_dim,
                time_dim=feature_repo.node_feature_dim(),
                output_dim=feature_repo.node_feature_dim(),
                n_head=n_head,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def _aggregate(
        self,
        layer_id: int,
        node_features: torch.Tensor,
        node_time_emb: torch.Tensor,
        edge_time_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_model = self._attention_models[layer_id - 1]
        output, weight = attention_model(
            node_features, node_time_emb, neighbor_emb, edge_time_emb, edge_features, mask
        )
        return output
