from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class Message(object):
    def __init__(self, ts: float, content: torch.Tensor) -> None:
        self.ts = ts  # TODO: clarify data type
        self.content = content

    def clone(self) -> Message:
        return Message(self.ts, self.content.clone())

    def detach(self) -> Message:
        return Message(self.ts, self.content.detach())


@dataclass
class MemorySnapshot:
    memory_tensor: torch.Tensor
    last_update: torch.Tensor
    messages: Optional[Dict[int, List[Message]]] = None


class Memory(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int) -> None:
        super(Memory, self).__init__()
        self._num_nodes = num_nodes
        self._memory_dim = memory_dim
        self.reset()

    def reset(self) -> None:
        self._memory_tensor = nn.Parameter(
            torch.zeros((self._num_nodes, self._memory_dim)), requires_grad=False
        )
        self._last_update = nn.Parameter(
            torch.zeros(self._num_nodes), requires_grad=False
        )
        self._messages: Dict[int, List[Message]] = defaultdict(list)

    def get_messages(self, nodes: torch.Tensor = None) -> Dict[int, List[Message]]:
        if nodes is None:
            return self._messages
        else:
            return {node_id: self._messages[node_id] for node_id in nodes.long().detach().numpy()}

    def store_messages(self, message_dict: Dict[int, List[Message]]) -> None:
        for node_id, messages in message_dict.items():
            self._messages[node_id] += messages

    def clear_messages(self, nodes: torch.Tensor) -> None:
        for node_id in nodes.long().detach().numpy():
            self._messages[node_id].clear()

    def get_memory_tensor(self, nodes: torch.Tensor = None) -> torch.Tensor:
        if nodes is None:
            return self._memory_tensor
        else:
            return self._memory_tensor[nodes, :]

    def set_memory_tensor(self, nodes: torch.Tensor, memory_tensor: torch.Tensor) -> None:
        self._memory_tensor[nodes, :] = memory_tensor

    def get_last_update(self, nodes: torch.Tensor = None) -> torch.Tensor:
        if nodes is None:
            return self._last_update
        else:
            return self._last_update[nodes]

    def set_last_update(self, nodes: torch.Tensor, values: torch.Tensor) -> None:
        self._last_update[nodes] = values

    def get_snapshot(self, require_message: bool = True) -> MemorySnapshot:
        return MemorySnapshot(
            memory_tensor=self._memory_tensor.data.clone(),
            last_update=self._last_update.data.clone(),
            messages=None if not require_message else {
                node_id: [msg.clone() for msg in messages]
                for node_id, messages in self._messages.items()
            },
        )

    def restore(self, memory_snapshot: MemorySnapshot) -> None:
        self._memory_tensor.data = memory_snapshot.memory_tensor.clone()
        self._last_update.data = memory_snapshot.last_update.clone()
        if memory_snapshot.memory_tensor is not None:
            self._messages = {
                node_id: [msg.clone() for msg in messages]
                for node_id, messages in memory_snapshot.messages.items()
            }

    def detach(self) -> None:
        self._memory_tensor.detach_()
        self._last_update.detach_()
        self._messages = {node_id: [msg.detach() for msg in messages] for node_id, messages in self._messages.items()}

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
