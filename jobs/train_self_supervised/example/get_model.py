import torch

from data.data import AbsFeatureRepo
from model.tgn.embedding_module import GraphAttentionEmbedding, SimpleEmbeddingModule
from model.tgn.memory_updater import GRUMemoryUpdater
from model.tgn.message_aggregator import LastMessageAggregator
from model.tgn.message_function import IdentityMessageFunction
from model.tgn.tgn import MemoryParams, TGN

EMBEDDING_DIM = MEMORY_DIM = 172  # Node feature dim
MESSAGE_DIM = 100


def get_model_no_memory(feature_repo: AbsFeatureRepo, device: torch.device) -> TGN:
    tgn = TGN(
        feature_repo, device,
        emb_module=SimpleEmbeddingModule(feature_repo, device, embedding_dim=EMBEDDING_DIM)
    )
    return tgn


def get_model_memory(feature_repo: AbsFeatureRepo, device: torch.device) -> TGN:
    tgn = TGN(
        feature_repo, device,
        emb_module=GraphAttentionEmbedding(
            feature_repo=feature_repo,
            device=device,
            embedding_dim=EMBEDDING_DIM,
            n_layers=1,
            n_neighbors=10,
            n_head=2,
            dropout=0.1
        ),
        memory_params=MemoryParams(
            memory_dim=MEMORY_DIM,
            message_dim=MESSAGE_DIM,
            message_function=IdentityMessageFunction(),
            message_aggregator=LastMessageAggregator(),
            memory_updater=GRUMemoryUpdater(memory_dim=MEMORY_DIM, message_dim=MESSAGE_DIM),
            update_memory_at_start=False,
            use_src_emb_in_message=False,
            use_dst_emb_in_message=False,
        )
    )
    return tgn
