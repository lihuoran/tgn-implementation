from typing import Any, Tuple

import torch
from torch import nn

from module import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        neighbor_emb_dim: int,
        time_dim: int,
        output_dim: int,
        n_head: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(TemporalAttentionLayer, self).__init__()

        self._query_dim = node_feat_dim + time_dim
        self._key_dim = self._value_dim = edge_feat_dim + neighbor_emb_dim + time_dim
        self._merge_layer = MergeLayer(
            input_dim_1=self._query_dim, input_dim_2=node_feat_dim, hidden_dim=node_feat_dim, output_dim=output_dim
        )

        self._attn = nn.MultiheadAttention(
            embed_dim=self._query_dim, kdim=self._key_dim, vdim=self._value_dim, num_heads=n_head, dropout=dropout
        )

    def forward(
        self,
        node_features: torch.Tensor,  # (B, node_feat_dim)
        node_time_emb: torch.Tensor,  # (B, 1, time_dim)
        neighbor_emb: torch.Tensor,  # (B, num_neighbor, node_feat_dim)
        edge_time_emb: torch.Tensor,  # (B, num_neighbor, time_dim)
        edge_features: torch.Tensor,  # (B, num_neighbor, edge_feat_dim)
        mask: torch.Tensor  # (B, num_neighbor)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_node_feat_unrolled = node_features.unsqueeze(dim=1)  # (B, 1, node_feat_dim)

        # qdim := node_feat_dim + edge_feat_dim
        # kdim := node_feat_dim + edge_feat_dim + time_dim
        query = torch.cat([src_node_feat_unrolled, node_time_emb], dim=2)  # (B, 1, qdim)
        key = torch.cat([neighbor_emb, edge_features, edge_time_emb], dim=2)  # (B, num_neighbor, kdim)

        query = query.permute([1, 0, 2])  # (1, B, qdim)
        key = key.permute([1, 0, 2])  # (num_neighbor, B, kdim)

        invalid_neighborhood_mask = torch.all(mask, dim=1, keepdim=True)  # (B, 1)
        mask[invalid_neighborhood_mask.squeeze(), 0] = False  # (B, num_neighbor)

        attn_output, attn_output_weights = self._attn(
            query=query, key=key, value=key, key_padding_mask=mask
        )  # (1, B, qdim), (B, 1, num_neighbor)

        attn_output = attn_output.squeeze()  # (B, qdim)
        attn_output_weights = attn_output_weights.squeeze()  # (B, num_neighbor)

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)  # (B, qdim)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)  # (B, num_neighbor)

        attn_output = self._merge_layer(attn_output, node_features)  # (B, output_dim)

        return attn_output, attn_output_weights  # (B, output_dim), (B, num_neighbor)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
