from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn

from model.tgn.memory import Memory, MemorySnapshot


class AbsMemoryUpdater(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMemoryUpdater, self).__init__()

    @abstractmethod
    def get_updated_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
        update_in_place: bool = True,
    ) -> MemorySnapshot:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class SequenceMemoryUpdater(AbsMemoryUpdater, metaclass=ABCMeta):
    def __init__(self):
        super(SequenceMemoryUpdater, self).__init__()

    def get_updated_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
        update_in_place: bool = True,
    ) -> MemorySnapshot:
        snapshot = memory.get_snapshot(requires_messages=False)
        snapshot.memory[unique_nodes] = self._update(unique_messages, snapshot.memory[unique_nodes])
        snapshot.last_update[unique_nodes] = unique_ts
        if update_in_place:
            memory.set_memory(unique_nodes, snapshot.memory[unique_nodes])
            memory.set_last_update(unique_nodes, snapshot.last_update[unique_nodes])
        return snapshot

    @abstractmethod
    def _update(self, unique_messages: torch.Tensor, memory_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory_dim: int, message_dim: int) -> None:
        super(GRUMemoryUpdater, self).__init__()
        self._gru_cell = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

    def _update(self, unique_messages: torch.Tensor, memory_tensor: torch.Tensor) -> torch.Tensor:
        return self._gru_cell(unique_messages, memory_tensor)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory_dim: int, message_dim: int) -> None:
        super(RNNMemoryUpdater, self).__init__()
        self._rnn_cell = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)

    def _update(self, unique_messages: torch.Tensor, memory_tensor: torch.Tensor) -> torch.Tensor:
        return self._rnn_cell(unique_messages, memory_tensor)
