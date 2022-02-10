import os
import shutil
from typing import Optional

import numpy as np
import torch
from torch import nn

from .workflow import WorkflowContext


class EarlyStopMonitor(object):
    def __init__(
        self,
        max_round: int = 3,
        higher_better: bool = True,
        tolerance: float = 1e-10,
    ) -> None:
        super(EarlyStopMonitor, self).__init__()

        self._max_round = max_round
        self._num_round = 0

        self._epoch_count = 0
        self._best_epoch = 0
        self._last_best: Optional[float] = None
        self._higher_better = higher_better
        self._tolerance = tolerance

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def num_round(self) -> int:
        return self._num_round

    def early_stop_check(self, curr_val: float) -> bool:
        self._epoch_count += 1

        if not self._higher_better:
            curr_val *= -1

        elif self._last_best is None or (curr_val - self._last_best) / np.abs(self._last_best) > self._tolerance:
            self._last_best = curr_val
            self._num_round = 0
            self._best_epoch = self._epoch_count
        else:  # No improvement
            self._num_round += 1

        return self._num_round >= self._max_round


class RandomNodeSelector(object):
    def __init__(self, nodes: np.ndarray, seed: int = None) -> None:
        self._ids = np.unique(nodes)
        self._total_size = len(self._ids)

        if seed is not None:
            self._seed = seed
            self._random_state = np.random.RandomState(self._seed)
        else:
            self._seed = None

    def sample(self, size: int) -> np.ndarray:
        indexes = np.random.randint(0, self._total_size, size) if self._seed is None \
            else self._random_state.randint(0, self._total_size, size)
        return self._ids[indexes]

    def reset_random_state(self) -> None:
        self._random_state = np.random.RandomState(self._seed)


def get_model_path(version_path: str, epoch: int = None) -> str:
    if epoch is not None:
        return os.path.join(version_path, 'saved_models', f'model_{epoch:05d}.ckpt')
    else:
        return os.path.join(version_path, 'saved_models', f'model_best.ckpt')


def save_model(workflow_context: WorkflowContext, model: nn.Module, version_path: str, epoch: int = None) -> None:
    path = get_model_path(version_path, epoch)
    workflow_context.logger.info(f'Save {type(model)} to {path}.')
    torch.save(model.state_dict(), path)


def load_model(workflow_context: WorkflowContext, model: nn.Module, version_path: str, epoch: int = None) -> None:
    path = get_model_path(version_path, epoch)
    workflow_context.logger.info(f'Load {type(model)} from {path}.')
    model.load_state_dict(torch.load(path))


def copy_best_model(version_path: str, best_epoch: int) -> None:
    shutil.copyfile(get_model_path(version_path, best_epoch), get_model_path(version_path))
