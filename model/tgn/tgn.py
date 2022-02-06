from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.data import AbsFeatureRepo, DataBatch
from model.abs_model import AbsModel
from model.tgn.embedding_module import AbsEmbeddingModule
from model.tgn.memory import Memory, MemorySnapshot, Message
from model.tgn.memory_updater import AbsMemoryUpdater
from model.tgn.merge_layer import MergeLayer
from model.tgn.message_aggregator import AbsMessageAggregator
from model.tgn.message_function import AbsMessageFunction
from module.time_encoder import TimeEncoder
from utils.training import NeighborFinder


@dataclass
class MemoryParams:
    memory_dim: int
    message_dim: int
    message_func: str = 'mlp'
    memory_update_at_start: bool = True
    use_src_emb_in_message: bool = False
    use_dst_emb_in_message: bool = False


class TGN(AbsModel):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        emb_module: AbsEmbeddingModule,
        memory_params: MemoryParams = None,
        neighbor_finder: NeighborFinder = None,
        device: str = 'cpu',
    ) -> None:
        super(TGN, self).__init__(feature_repo)

        self._use_memory = memory_params is not None
        self._memory_params = memory_params
        if self._use_memory:
            self._init_memory()

        self._device = device

        self._emb_module = emb_module

        self._neighbor_finder = neighbor_finder

        self._affinity_score = MergeLayer(
            input_dim_1=self._emb_module.embedding_dim,
            input_dim_2=self._emb_module.embedding_dim,
            hidden_dim=256,
            output_dim=1  # TODO: remove constant numbers
        )

    def train_mode(self) -> None:
        self.train()
        self._emb_module.train()

    def eval_mode(self) -> None:
        self.eval()
        self._emb_module.eval()

    def _init_memory(self) -> None:
        self._time_encoder = TimeEncoder(self._node_feature_dim)

        # if self._memory_params.message_func == 'identity':
        #     raw_message_dim = self._memory_params.message_dim
        # else:
        #     raw_message_dim = 2 * self._memory_params.memory_dim + self._edge_feat_dim + self._time_encoder.dimension

        self._memory = Memory(num_nodes=self._num_nodes, memory_dim=self._memory_params.memory_dim)
        self._memory.to(self._device)

        self._message_aggregator = AbsMessageAggregator()  # TODO
        self._message_function = AbsMessageFunction()  # TODO
        self._memory_updater = AbsMemoryUpdater()  # TODO

    def _compute_temporal_embeddings_with_memory(
        self, batch: DataBatch, n_neighbors: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # TODO

    def _compute_temporal_embeddings_without_memory(
        self, batch: DataBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_emb = self._emb_module.compute_embedding(batch.src_ids, batch.timestamps)
        dst_emb = self._emb_module.compute_embedding(batch.dst_ids, batch.timestamps)
        return src_emb, dst_emb

    def compute_temporal_embeddings(self, batch: DataBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_memory:
            return self._compute_temporal_embeddings_with_memory(batch)
        else:
            return self._compute_temporal_embeddings_without_memory(batch)

    def compute_edge_probabilities(self, batch: DataBatch) -> torch.Tensor:
        src_emb, dst_emb = self.compute_temporal_embeddings(batch)  # (B, emb_dim), (B, emb_dim)
        score = self._affinity_score(src_emb, dst_emb).squeeze()  # (B,)
        return score.sigmoid()  # (B,)

    def _update_memory(self, nodes: torch.Tensor, messages: Dict[int, List[Message]]) -> None:
        unique_nodes, unique_messages, unique_ts = self._message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self._message_function.compute_message(unique_messages)
        self._memory_updater.update_memory(self._memory, unique_nodes, unique_messages, unique_ts)

    def _get_updated_memory(
        self, nodes: torch.Tensor, messages: Dict[int, List[Message]]
    ) -> MemorySnapshot:
        unique_nodes, unique_messages, unique_ts = self._message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self._message_function.compute_message(unique_messages)
        return self._memory_updater.get_updated_memory(self._memory, unique_nodes, unique_messages, unique_ts)

    def _get_raw_message(
        self, batch: DataBatch, src_emb: torch.Tensor, dst_emb: torch.Tensor
    ) -> Tuple[np.ndarray, Dict[int, List[Message]]]:
        ts = torch.from_numpy(batch.timestamps).float().to(self._device)
        edge_feat = torch.from_numpy(batch.edge_ids).float().to(self._device)

        src_memory = src_emb if self._memory_params.use_src_emb_in_message else self._memory.get_memory(batch.src_ids)
        dst_memory = dst_emb if self._memory_params.use_dst_emb_in_message else self._memory.get_memory(batch.dst_ids)

        src_time_delta = ts - self._memory.get_last_update(batch.src_ids)
        src_time_delta_encoding = self._time_encoder(src_time_delta.unsqueeze(dim=1)).view(batch.size, -1)

        src_message = torch.cat([src_memory, dst_memory, edge_feat, src_time_delta_encoding], dim=1)

        messages: Dict[int, List[Message]] = defaultdict(list)
        unique_src_ids = np.unique(batch.src_ids)
        for src_id, message, t in zip(batch.src_ids, src_message, ts):
            messages[src_id].append(Message(t, message))

        return unique_src_ids, messages

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._neighbor_finder = neighbor_finder
        self._emb_module.set_neighbor_finder(neighbor_finder)
