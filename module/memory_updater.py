from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class AbsMemoryUpdater(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsMemoryUpdater, self).__init__()

    @abstractmethod
    def update_memory(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_updated_memory(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
