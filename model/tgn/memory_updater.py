from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import torch
from torch import nn

from model.tgn.memory import Memory, MemorySnapshot


class AbsMemoryUpdater(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMemoryUpdater, self).__init__()

    @abstractmethod
    def update_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_updated_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class SequenceMemoryUpdater(AbsMemoryUpdater, metaclass=ABCMeta):
    def __init__(self):
        super(SequenceMemoryUpdater, self).__init__()

    def update_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
    ) -> None:
        if len(unique_nodes) <= 0:
            return

        memory_tensor = memory.get_memory_tensor(unique_nodes)
        memory.set_last_update(unique_nodes, unique_ts)

        updated_memory_tensor = self._update(unique_messages, memory_tensor)
        memory.set_memory_tensor(unique_nodes, updated_memory_tensor)

    def get_updated_memory(
        self,
        memory: Memory,
        unique_nodes: torch.Tensor,
        unique_messages: torch.Tensor,
        unique_ts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        updated_memory_tensor = memory.get_memory_tensor().data.clone()
        updated_last_update = memory.get_last_update().data.clone()

        if len(unique_nodes) > 0:
            updated_memory_tensor[unique_nodes] = self._update(unique_messages, updated_memory_tensor[unique_nodes])
            updated_last_update[unique_nodes] = unique_ts

        return updated_memory_tensor, updated_last_update

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
