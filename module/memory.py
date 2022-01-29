from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import nn


class Message(object):
    def __init__(self, ts: float, content: Union[np.ndarray, torch.Tensor]) -> None:
        self.ts = ts  # TODO: clarity data type
        self.content = content  # TODO: clarity data type

    def clone(self) -> Message:
        return Message(self.ts, self.content.clone())


@dataclass
class MemorySnapshot:
    memory: torch.Tensor
    last_update: torch.Tensor
    messages: Dict[int, List[Message]]


class Memory(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        aggr_method: str = 'sum',
    ) -> None:
        super(Memory, self).__init__()
        self._num_nodes = num_nodes
        self._memory_dim = memory_dim
        self._aggr_method = aggr_method

        self._memory = nn.Parameter(
            torch.zeros((self._num_nodes, self._memory_dim)), requires_grad=False
        )
        self._last_update = nn.Parameter(
            torch.zeros(self._num_nodes), requires_grad=False
        )
        self._messages: Dict[int, List[Message]] = defaultdict(list)

    def store_messages(self, message_dict: Dict[int, List[Message]]) -> None:
        for node_id, messages in message_dict.items():
            self._messages[node_id] += messages

    def clear_messages(self, nodes: np.ndarray) -> None:
        for node_id in nodes:
            self._messages[node_id].clear()

    def get_memory(self, nodes: np.ndarray) -> torch.Tensor:
        return self._memory[nodes, :]

    def set_memory(self, nodes: np.ndarray, values: torch.Tensor) -> None:
        self._memory[nodes, :] = values

    def get_last_update(self, nodes: np.ndarray) -> torch.Tensor:
        return self._last_update[nodes]

    @property
    def snapshot(self) -> MemorySnapshot:
        messages_clone = {
            node_id: [msg.clone() for msg in messages]
            for node_id, messages in self._messages.items()
        }
        return MemorySnapshot(
            memory=self._memory.data.clone(),
            last_update=self._last_update.data.clone(),
            messages=messages_clone,
        )

    def restore(self, memory_snapshot: MemorySnapshot) -> None:
        self._memory.data = memory_snapshot.memory.clone()
        self._last_update.data = memory_snapshot.last_update.clone()
        self._messages = {
            node_id: [msg.clone() for msg in messages]
            for node_id, messages in memory_snapshot.messages.items()
        }

    def detach(self) -> None:
        raise NotImplementedError  # TODO

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
