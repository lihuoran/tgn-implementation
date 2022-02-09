import collections
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from data import Dataset


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

    def early_stop_check(self, curr_val: float) -> bool:
        self._epoch_count += 1

        if not self._higher_better:
            curr_val *= -1

        if self._last_best is None:
            self._last_best = curr_val

        elif (curr_val - self._last_best) / np.abs(self._last_best) > self._tolerance:  # Has improvement
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


class NeighborFinder:
    def __init__(
        self,
        adj_lists: Dict[int, List[Tuple[float, int, int]]],
        uniform: bool = False,
        seed: int = None
    ) -> None:
        self._adj_lists: Dict[int, List[Tuple[float, int, int]]] = {
            k: sorted(v, key=lambda x: x[0]) for k, v in adj_lists.items()
        }
        self._ts_lists: Dict[int, List[float]] = {
            k: [x[0] for x in v] for k, v in self._adj_lists.items()
        }

        self._uniform = uniform
        if seed is not None:
            self._seed = seed
            self._random_state = np.random.RandomState(self._seed)
        else:
            self._seed = None

    def _find_before(self, src_id: int, cut_time: float) -> List[Tuple[float, int, int]]:
        i = np.searchsorted(self._ts_lists[src_id], cut_time)
        return self._adj_lists[src_id][:i]

    def get_temporal_neighbor(
        self,
        nodes: np.ndarray,
        timestamps: np.ndarray,
        k: int = 20
    ) -> List[List[Tuple[float, int, int]]]:  # (timestamp, edge_id, node_id)
        assert (len(nodes) == len(timestamps))
        assert k > 0

        temporal_neighbor = []
        for i, (src_id, ts) in enumerate(zip(nodes, timestamps)):
            neighbors = self._find_before(src_id, ts)
            if len(neighbors) == 0:
                ngh_list = [(0.0, 0, 0)] * k  # Empty result
            elif self._uniform:
                sampled_idx: List[int] = sorted(np.random.randint(0, len(neighbors), k))
                ngh_list = [neighbors[idx] for idx in sampled_idx]  # Keep the original order
            else:
                ngh_list = neighbors[-k:]
                if len(ngh_list) < k:  # Left padding
                    ngh_list = [(0.0, 0, 0)] * (k - len(ngh_list)) + ngh_list

            assert len(ngh_list) == k
            temporal_neighbor.append(ngh_list)
        return temporal_neighbor

    def reset_random_state(self) -> None:
        self._random_state = np.random.RandomState(self._seed)


def get_model_path(version_path: str, epoch: int = None) -> str:
    if epoch is not None:
        return os.path.join(version_path, 'saved_models', f'model_{epoch:05d}.ckpt')
    else:
        return os.path.join(version_path, 'saved_models', f'model_best.ckpt')


def save_model(model: nn.Module, version_path: str, epoch: int = None) -> None:
    torch.save(model.state_dict(), get_model_path(version_path, epoch))


def load_model(model: nn.Module, version_path: str, epoch: int = None) -> None:
    model.load_state_dict(torch.load(get_model_path(version_path, epoch)))


def copy_best_model(version_path: str, best_epoch: int) -> None:
    shutil.copyfile(get_model_path(version_path, best_epoch), get_model_path(version_path))


def get_neighbor_finder(data: Dataset, uniform: bool) -> NeighborFinder:
    adj_list: Dict[int, List[Tuple[float, int, int]]] = collections.defaultdict(list)

    data_iter = zip(data.src_ids, data.dst_ids, data.edge_ids, data.timestamps)
    for src_id, dst_id, edge_id, timestamp in data_iter:
        adj_list[int(src_id)].append((timestamp, edge_id, dst_id))
        adj_list[int(dst_id)].append((timestamp, edge_id, src_id))

    return NeighborFinder(adj_list, uniform=uniform)
