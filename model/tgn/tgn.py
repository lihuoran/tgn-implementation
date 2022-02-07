from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from data.data import AbsFeatureRepo, DataBatch
from model.abs_model import AbsModel
from model.tgn.embedding_module import AbsEmbeddingModule
from model.tgn.memory import Memory, MemorySnapshot, Message
from model.tgn.memory_updater import AbsMemoryUpdater
from model.tgn.message_aggregator import AbsMessageAggregator
from model.tgn.message_function import AbsMessageFunction
from module.merge_layer import MergeLayer
from module.time_encoder import TimeEncoder
from utils.training import NeighborFinder


@dataclass
class MemoryParams:
    memory_dim: int
    message_dim: int

    message_function: AbsMessageFunction
    message_aggregator: AbsMessageAggregator
    memory_updater: AbsMemoryUpdater

    update_memory_at_start: bool = True
    use_src_emb_in_message: bool = False
    use_dst_emb_in_message: bool = False

    src_time_shift_mean: float = 0.0
    src_time_shift_std: float = 1
    dst_time_shift_mean: float = 0.0
    dst_time_shift_std: float = 1


def _normalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (data - mean) / std


class TGN(AbsModel):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        emb_module: AbsEmbeddingModule,
        memory_params: MemoryParams = None,
        neighbor_finder: NeighborFinder = None,
    ) -> None:
        super(TGN, self).__init__(feature_repo, device)

        self._use_memory = memory_params is not None
        self._memory_params = memory_params
        if self._use_memory:
            self._init_memory()

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

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._emb_module.set_neighbor_finder(neighbor_finder)

    def _init_memory(self) -> None:
        self._time_encoder = TimeEncoder(self._node_feature_dim)

        self._memory = Memory(num_nodes=self._num_nodes, memory_dim=self._memory_params.memory_dim)

        self._message_function = self._memory_params.message_function
        self._message_aggregator = self._memory_params.message_aggregator
        self._memory_updater = self._memory_params.memory_updater

    def _compute_temporal_embeddings_with_memory(self, batch: DataBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        all_nodes = torch.arange(self._feature_repo.num_nodes()).long()
        if self._memory_params.update_memory_at_start:
            snapshot = self._get_updated_memory(all_nodes, self._memory.get_messages(), update_in_place=False)
            memory_tensor, last_update = snapshot.memory_tensor, snapshot.last_update
        else:
            memory_tensor = self._memory.get_memory_tensor(all_nodes)
            last_update = self._memory.get_last_update(all_nodes)

        src_emb = self._emb_module.compute_embedding(
            nodes=batch.src_ids, timestamps=batch.timestamps, memory_tensor=memory_tensor,
            time_diffs=_normalize(
                data=batch.timestamps - last_update[batch.src_ids],
                mean=self._memory_params.src_time_shift_mean,
                std=self._memory_params.src_time_shift_std,
            )
        )
        dst_emb = self._emb_module.compute_embedding(
            nodes=batch.dst_ids, timestamps=batch.timestamps, memory_tensor=memory_tensor,
            time_diffs=_normalize(
                data=batch.timestamps - last_update[batch.dst_ids],
                mean=self._memory_params.dst_time_shift_mean,
                std=self._memory_params.dst_time_shift_std,
            )
        )

        if self._memory_params.update_memory_at_start:
            self._get_updated_memory(batch.src_ids, self._memory.get_messages(), update_in_place=True)
            self._get_updated_memory(batch.dst_ids, self._memory.get_messages(), update_in_place=True)
            assert torch.allclose(memory_tensor[batch.src_ids], self._memory.get_memory_tensor(batch.src_ids))
            assert torch.allclose(memory_tensor[batch.dst_ids], self._memory.get_memory_tensor(batch.dst_ids))
            self._memory.clear_messages(batch.src_ids)
            self._memory.clear_messages(batch.dst_ids)

        src_messages = self._get_raw_message(batch, src_emb, dst_emb, reverse=False)
        dst_messages = self._get_raw_message(batch, src_emb, dst_emb, reverse=True)
        if self._memory_params.update_memory_at_start:
            self._memory.store_messages(src_messages)
            self._memory.store_messages(dst_messages)
        else:
            self._get_updated_memory(torch.Tensor(list(src_messages.keys())).long(), src_messages, update_in_place=True)
            self._get_updated_memory(torch.Tensor(list(dst_messages.keys())).long(), dst_messages, update_in_place=True)

        # if self.dyrep: ...  # TODO
        return src_emb, dst_emb

    def _compute_temporal_embeddings_without_memory(self, batch: DataBatch) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _get_updated_memory(
        self, nodes: torch.Tensor, messages: Dict[int, List[Message]], update_in_place: bool = True
    ) -> MemorySnapshot:
        unique_nodes, unique_messages, unique_ts = self._message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self._message_function.compute_message(unique_messages)

        return self._memory_updater.get_updated_memory(
            self._memory, unique_nodes, unique_messages, unique_ts, update_in_place
        )

    def _get_raw_message(
        self, batch: DataBatch, src_emb: torch.Tensor, dst_emb: torch.Tensor, reverse: bool
    ) -> Dict[int, List[Message]]:
        if not reverse:
            src_ids, dst_ids = batch.src_ids, batch.dst_ids
        else:
            src_ids, dst_ids = batch.dst_ids, batch.src_ids
            src_emb, dst_emb = dst_emb, src_emb
        edge_ids = batch.edge_ids.detach().numpy()
        ts = batch.timestamps

        edge_feat = torch.from_numpy(self._feature_repo.get_edge_feature(edge_ids)).float().to(self._device)
        src_memory = src_emb if self._memory_params.use_src_emb_in_message else self._memory.get_memory_tensor(src_ids)
        dst_memory = dst_emb if self._memory_params.use_dst_emb_in_message else self._memory.get_memory_tensor(dst_ids)

        src_time_delta = ts - self._memory.get_last_update(src_ids)
        src_time_delta_encoding = self._time_encoder(src_time_delta.unsqueeze(dim=1)).view(batch.size, -1)

        src_message = torch.cat([src_memory, dst_memory, edge_feat, src_time_delta_encoding], dim=1)

        messages: Dict[int, List[Message]] = defaultdict(list)
        for src_id, message, t in zip(src_ids, src_message, ts):
            messages[src_id].append(Message(t, message))

        return messages
    #
    # def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
    #     self._neighbor_finder = neighbor_finder
    #     self._emb_module.set_neighbor_finder(neighbor_finder)
