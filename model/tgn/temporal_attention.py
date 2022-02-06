from typing import Any, Tuple

import torch
from torch import nn

from model.tgn.merge_layer import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        ngh_feat_dim: int,
        time_dim: int,
        output_dim: int,
        n_head: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(TemporalAttentionLayer, self).__init__()

        self._query_dim = node_feat_dim + time_dim
        self._key_dim = self._value_dim = edge_feat_dim + ngh_feat_dim + time_dim
        self._merge_layer = MergeLayer(
            input_dim_1=self._query_dim, input_dim_2=node_feat_dim, hidden_dim=node_feat_dim, output_dim=output_dim
        )

        self._attn = nn.MultiheadAttention(
            embed_dim=self._query_dim, kdim=self._key_dim, vdim=self._value_dim, num_heads=n_head, dropout=dropout
        )

    def forward(
        self,
        src_node_feat: torch.Tensor,    # (B, node_feat_dim)
        src_time_feat: torch.Tensor,    # (B, 1, time_dim)
        ngh_feat: torch.Tensor,         # (B, num_ngh, node_feat_dim)
        ngh_time_feat: torch.Tensor,    # (B, num_ngh, time_dim)
        edge_feat: torch.Tensor,        # (B, num_ngh, edge_feat_dim)
        ngh_padding_mask: torch.Tensor  # (B, num_ngh)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_node_feat_unrolled = src_node_feat.unsqueeze(dim=1)  # (B, 1, node_feat_dim)

        # qdim = node_feat_dim + edge_feat_dim
        # kdim = node_feat_dim + edge_feat_dim + time_dim
        query = torch.cat([src_node_feat_unrolled, src_time_feat], dim=2)  # (B, 1, qdim)
        key = torch.cat([ngh_feat, edge_feat, ngh_time_feat], dim=2)  # (B, num_ngh, kdim)

        query = query.permute([1, 0, 2])  # (1, B, qdim)
        key = key.permute([1, 0, 2])  # (num_ngh, B, kdim)

        invalid_neighborhood_mask = torch.all(ngh_padding_mask, dim=1, keepdim=True)  # (B, 1)
        ngh_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False  # (B, num_ngh)

        attn_output, attn_output_weights = self._attn(
            query=query, key=key, value=key, key_padding_mask=ngh_padding_mask
        )  # (1, B, qdim), (B, 1, num_ngh)

        attn_output = attn_output.squeeze()  # (B, qdim)
        attn_output_weights = attn_output_weights.squeeze()  # (B, num_ngh)

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)  # (B, qdim)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)  # (B, num_ngh)

        attn_output = self._merge_layer(attn_output, src_node_feat)  # (B, output_dim)

        return attn_output, attn_output_weights  # (B, output_dim), (B, num_ngh)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
