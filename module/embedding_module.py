from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class AbsEmbeddingModule(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsEmbeddingModule, self).__init__()

    @abstractmethod
    def compute_embedding(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
