from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from data import AbsFeatureRepo, DataBatch
from model import AbsModel, EmbeddingBundle
from module import MergeLayer, TimeEncoder
from utils import NeighborFinder
from .embedding_module import AbsEmbeddingModule
from .memory import Memory, Message
from .memory_updater import AbsMemoryUpdater
from .message_aggregator import AbsMessageAggregator
from .message_function import AbsMessageFunction


@dataclass
class MemoryParams:
    memory_dim: int

    message_function: AbsMessageFunction
    message_aggregator: AbsMessageAggregator
    memory_updater: AbsMemoryUpdater

    update_memory_at_start: bool = True
    use_src_emb_in_message: bool = False
    use_dst_emb_in_message: bool = False

    src_time_shift_mean: float = 0.0
    src_time_shift_std: float = 1.0
    dst_time_shift_mean: float = 0.0
    dst_time_shift_std: float = 1.0


def _normalize(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (data - mean) / std


class TGN(AbsModel):
    def __init__(
        self,
        feature_repo: AbsFeatureRepo,
        device: torch.device,
        emb_module: AbsEmbeddingModule,
        memory_params: MemoryParams = None,
    ) -> None:
        super(TGN, self).__init__(feature_repo, device)

        self._use_memory = memory_params is not None
        self._memory_params = memory_params
        if self._use_memory:
            self._init_memory()

        self._emb_module = emb_module

        self._affinity_score = MergeLayer(
            input_dim_1=self._emb_module.embedding_dim,
            input_dim_2=self._emb_module.embedding_dim,
            hidden_dim=256,
            output_dim=1  # TODO: remove constant numbers
        )

    def epoch_start_step(self) -> None:
        if self._use_memory:
            self._memory.reset()

    def backward_post_step(self) -> None:
        if self._use_memory:
            self._memory.detach()

    def set_neighbor_finder(self, neighbor_finder: NeighborFinder) -> None:
        self._emb_module.set_neighbor_finder(neighbor_finder)

    def _init_memory(self) -> None:
        self._time_encoder = TimeEncoder(self._node_feature_dim)

        self._memory = Memory(num_nodes=self._num_nodes, memory_dim=self._memory_params.memory_dim)

        self._message_function = self._memory_params.message_function
        self._message_aggregator = self._memory_params.message_aggregator
        self._memory_updater = self._memory_params.memory_updater

    def _compute_temporal_embeddings_with_memory(self, batch: DataBatch) -> EmbeddingBundle:
        if self._memory_params.update_memory_at_start:
            memory_tensor, last_update = self._get_updated_memory(self._memory.get_messages())
        else:
            memory_tensor = self._memory.get_memory_tensor()
            last_update = self._memory.get_last_update()

        src_time_diffs = _normalize(
            data=batch.timestamps - last_update[batch.src_ids],
            mean=self._memory_params.src_time_shift_mean,
            std=self._memory_params.src_time_shift_std,
        )
        dst_time_diffs = _normalize(
            data=batch.timestamps - last_update[batch.dst_ids],
            mean=self._memory_params.dst_time_shift_mean,
            std=self._memory_params.dst_time_shift_std,
        )
        if batch.neg_ids is not None:
            neg_time_diffs = _normalize(
                data=batch.timestamps - last_update[batch.neg_ids],
                mean=self._memory_params.dst_time_shift_mean,
                std=self._memory_params.dst_time_shift_std,
            )
            src_emb, dst_emb, neg_emb = self._compute_temporal_embeddings(
                batch=batch, memory_tensor=memory_tensor, time_diffs=[src_time_diffs, dst_time_diffs, neg_time_diffs]
            )
        else:
            src_emb, dst_emb, neg_emb = self._compute_temporal_embeddings(
                batch=batch, memory_tensor=memory_tensor, time_diffs=[src_time_diffs, dst_time_diffs]
            )

        if self._memory_params.update_memory_at_start:
            self._update_memory(self._memory.get_messages(), batch.src_ids)
            self._update_memory(self._memory.get_messages(), batch.dst_ids)
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
            self._update_memory(src_messages)
            self._update_memory(dst_messages)

        # if self.dyrep: ...  # TODO
        return src_emb, dst_emb, neg_emb

    def _compute_temporal_embeddings(
        self, batch: DataBatch, memory_tensor: torch.Tensor = None, time_diffs: List[torch.Tensor] = None
    ) -> EmbeddingBundle:
        batch_size = batch.size
        if batch.neg_ids is not None:
            tot_ids = torch.cat([batch.src_ids, batch.dst_ids, batch.neg_ids])
            tot_ts = torch.cat([batch.timestamps, batch.timestamps, batch.timestamps])
            tot_time_diff = None if time_diffs is None else torch.cat(time_diffs)
            tot_emb = self._emb_module.compute_embedding(
                nodes=tot_ids, timestamps=tot_ts, memory_tensor=memory_tensor, time_diffs=tot_time_diff
            )
            return tot_emb[:batch_size, :], tot_emb[batch_size:batch_size * 2, :], tot_emb[batch_size * 2:, :]
        else:
            tot_ids = torch.cat([batch.src_ids, batch.dst_ids])
            tot_ts = torch.cat([batch.timestamps, batch.timestamps])
            tot_time_diff = None if time_diffs is None else torch.cat(time_diffs)
            tot_emb = self._emb_module.compute_embedding(
                nodes=tot_ids, timestamps=tot_ts, memory_tensor=memory_tensor, time_diffs=tot_time_diff
            )
            return tot_emb[:batch_size, :], tot_emb[batch_size:, :], None

    def compute_temporal_embeddings(self, batch: DataBatch) -> EmbeddingBundle:
        if self._use_memory:
            return self._compute_temporal_embeddings_with_memory(batch)
        else:
            return self._compute_temporal_embeddings(batch)

    def compute_edge_probabilities(self, batch: DataBatch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(batch)  # (B, emb_dim), (B, emb_dim), (B, emb_dim)
        pos_score = self._affinity_score(src_emb, dst_emb).squeeze()  # (B,)
        neg_score = None if neg_emb is None else self._affinity_score(src_emb, neg_emb).squeeze()  # (B,)
        return pos_score.sigmoid(), None if neg_score is None else neg_score.sigmoid()  # (B,), (B,)

    def _aggregate_and_compute_message(
        self, nodes: torch.Tensor, messages: Dict[int, List[Message]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.tensor]:
        unique_nodes, unique_messages, unique_ts = self._message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self._message_function.compute_message(unique_messages)

        return unique_nodes, unique_messages, unique_ts

    def _update_memory(self, messages: Dict[int, List[Message]], nodes: torch.Tensor = None) -> None:
        if nodes is None:
            nodes = torch.Tensor(list(messages.keys())).long()
        unique_nodes, unique_messages, unique_ts = self._aggregate_and_compute_message(nodes, messages)
        self._memory_updater.update_memory(self._memory, unique_nodes, unique_messages, unique_ts)

    def _get_updated_memory(
        self, messages: Dict[int, List[Message]], nodes: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if nodes is None:
            nodes = torch.Tensor(list(messages.keys())).long()
        unique_nodes, unique_messages, unique_ts = self._aggregate_and_compute_message(nodes, messages)
        return self._memory_updater.get_updated_memory(self._memory, unique_nodes, unique_messages, unique_ts)

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
        for i in range(len(src_ids)):
            messages[int(src_ids[i])].append(Message(int(ts[i]), src_message[i]))

        return messages
