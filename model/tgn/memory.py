from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class Message(object):
    def __init__(self, ts: float, content: torch.Tensor) -> None:
        self.ts = ts  # TODO: clarity data type
        self.content = content

    def clone(self) -> Message:
        return Message(self.ts, self.content.clone())


@dataclass
class MemorySnapshot:
    memory: torch.Tensor
    last_update: torch.Tensor
    messages: Optional[Dict[int, List[Message]]] = None


class Memory(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int) -> None:
        super(Memory, self).__init__()
        self._num_nodes = num_nodes
        self._memory_dim = memory_dim

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

    def clear_messages(self, nodes: torch.Tensor) -> None:
        for node_id in nodes.long().detach().numpy():
            self._messages[node_id].clear()

    def get_memory(self, nodes: torch.Tensor) -> torch.Tensor:
        return self._memory[nodes]

    def set_memory(self, nodes: torch.Tensor, values: torch.Tensor) -> None:
        self._memory[nodes] = values

    def get_last_update(self, nodes: torch.Tensor) -> torch.Tensor:
        return self._last_update[nodes]

    def set_last_update(self, nodes: torch.Tensor, values: torch.Tensor) -> None:
        self._last_update[nodes] = values

    def get_snapshot(self, requires_messages: bool = True) -> MemorySnapshot:
        return MemorySnapshot(
            memory=self._memory.clone(),
            last_update=self._last_update.clone(),
            messages=None if not requires_messages else {
                node_id: [msg.clone() for msg in messages]
                for node_id, messages in self._messages.items()
            },
        )

    def restore(self, memory_snapshot: MemorySnapshot) -> None:
        self._memory = memory_snapshot.memory.clone()
        self._last_update = memory_snapshot.last_update.clone()
        if memory_snapshot.memory is not None:
            self._messages = {
                node_id: [msg.clone() for msg in messages]
                for node_id, messages in memory_snapshot.messages.items()
            }

    def detach(self) -> None:
        raise NotImplementedError  # TODO

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
