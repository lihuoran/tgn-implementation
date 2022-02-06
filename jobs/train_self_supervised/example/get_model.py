from data.data import AbsFeatureRepo
from model.tgn.tgn import TGN
from model.tgn.embedding_module import SimpleEmbeddingModule


def get_model(feature_repo: AbsFeatureRepo) -> TGN:
    tgn = TGN(
        feature_repo,
        emb_module=SimpleEmbeddingModule(num_nodes=feature_repo.num_nodes(), embedding_dim=128)
    )
    return tgn
