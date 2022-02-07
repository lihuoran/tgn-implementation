from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from model.tgn.memory import Message


class AbsMessageAggregator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMessageAggregator, self).__init__()

    @abstractmethod
    def aggregate(
        self,
        nodes: torch.Tensor,
        messages: Dict[int, List[Message]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            nodes: np.ndarray
                shape = (batch_size,)
            messages: Dict[int, List[Message]]

        Returns:
            unique_nodes: np.ndarray
            unique_message: torch.Tensor
            unique_ts: torch.Tensor
        """
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class LastMessageAggregator(AbsMessageAggregator):
    def __init__(self) -> None:
        super(LastMessageAggregator, self).__init__()

    def aggregate(
        self,
        nodes: torch.Tensor,
        messages: Dict[int, List[Message]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_nodes = torch.unique(nodes)
        unique_message_contents = []
        unique_ts = []
        to_update_nodes = []

        for node_id in unique_nodes:
            message_list = messages[node_id]
            if len(message_list) > 0:
                to_update_nodes.append(node_id)
                unique_message_contents.append(message_list[-1].content)
                unique_ts.append(message_list[-1].ts)

        unique_message_contents = torch.stack(unique_message_contents) if len(to_update_nodes) > 0 else []
        unique_ts = torch.Tensor(unique_ts).float() if len(to_update_nodes) > 0 else []
        return torch.Tensor(to_update_nodes).long(), unique_message_contents, unique_ts


class MeanMessageAggregator(AbsMessageAggregator):
    def __init__(self) -> None:
        super(MeanMessageAggregator, self).__init__()

    def aggregate(
        self,
        nodes: torch.Tensor,
        messages: Dict[int, List[Message]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_nodes = torch.unique(nodes)
        unique_message_contents = []
        unique_ts = []
        to_update_nodes = []

        for node_id in unique_nodes:
            message_list = messages[node_id]
            if len(message_list) > 0:
                to_update_nodes.append(node_id)
                unique_message_contents.append(torch.mean(torch.stack([msg.content for msg in message_list], dim=0)))
                unique_ts.append(message_list[-1].ts)

        unique_message_contents = torch.stack(unique_message_contents) if len(to_update_nodes) > 0 else []
        unique_ts = torch.Tensor(unique_ts).float() if len(to_update_nodes) > 0 else []
        return torch.Tensor(to_update_nodes).long(), unique_message_contents, unique_ts
