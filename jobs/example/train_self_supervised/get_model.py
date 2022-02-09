import torch

from data import AbsFeatureRepo
from model.tgn import (
    GraphAttentionEmbedding, GRUMemoryUpdater, IdentityMessageFunction, LastMessageAggregator, MemoryParams,
    SimpleEmbeddingModule, TGN,
)

MESSAGE_DIM = 100


def get_model_no_memory(feature_repo: AbsFeatureRepo, device: torch.device) -> TGN:
    tgn = TGN(
        feature_repo, device,
        emb_module=SimpleEmbeddingModule(feature_repo, device, embedding_dim=feature_repo.node_feature_dim())
    )
    return tgn


def get_model_memory(feature_repo: AbsFeatureRepo, device: torch.device) -> TGN:
    raw_message_dim = 3 * feature_repo.node_feature_dim() + feature_repo.edge_feature_dim()
    tgn = TGN(
        feature_repo=feature_repo, device=device,
        emb_module=GraphAttentionEmbedding(
            feature_repo=feature_repo, device=device,
            embedding_dim=feature_repo.node_feature_dim(),
            n_layers=1, n_neighbors=10,
            n_head=2, dropout=0.1
        ),
        memory_params=MemoryParams(
            memory_dim=feature_repo.node_feature_dim(),
            message_function=IdentityMessageFunction(raw_message_dim=raw_message_dim),
            message_aggregator=LastMessageAggregator(),
            memory_updater=GRUMemoryUpdater(memory_dim=feature_repo.node_feature_dim(), message_dim=raw_message_dim),
            update_memory_at_start=False,
            use_src_emb_in_message=False,
            use_dst_emb_in_message=False,
        ),
    )
    return tgn
