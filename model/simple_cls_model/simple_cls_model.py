import torch

from model import AbsBinaryClassificationModel
from module import MLP


class SimpleClassificationModel(AbsBinaryClassificationModel):
    def __init__(self, input_dim: int, device: torch.device) -> None:
        super(SimpleClassificationModel, self).__init__(input_dim, device)
        self._mlp = MLP(input_dim)

    def get_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)
