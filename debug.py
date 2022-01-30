import torch

from module.embedding_module import SimpleEmbeddingModule

emb = SimpleEmbeddingModule(10, 5)
nodes = torch.Tensor([1, 2, 3]).long()
res = emb.compute_embedding(nodes, torch.ones(1))
print(res, res.requires_grad)