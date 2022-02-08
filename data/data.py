from __future__ import annotations

import math
import random
from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Generator, Set, Tuple

import numpy as np
import pandas as pd
import torch


class AbsFeatureRepo(object):
    def __init__(self) -> None:
        super(AbsFeatureRepo, self).__init__()

    @abstractmethod
    def get_node_feature(self, nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_edge_feature(self, edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def num_edges(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def node_feature_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def edge_feature_dim(self) -> int:
        raise NotImplementedError


class StaticFeatureRepo(AbsFeatureRepo):
    """Used during training, since the data will not be changed during training.
    """
    def __init__(self, node_feature: np.ndarray, edge_feature: np.ndarray) -> None:
        super(StaticFeatureRepo, self).__init__()

        assert len(node_feature.shape) == 2
        assert len(edge_feature.shape) == 2

        self._node_feature = node_feature
        self._edge_feature = edge_feature

    def get_node_feature(self, nodes: np.ndarray) -> np.ndarray:
        return self._node_feature[nodes]

    def get_edge_feature(self, edges: np.ndarray) -> np.ndarray:
        return self._edge_feature[edges]

    def num_nodes(self) -> int:
        return self._node_feature.shape[0]

    def num_edges(self) -> int:
        return self._edge_feature.shape[0]

    def node_feature_dim(self) -> int:
        return self._node_feature.shape[1]

    def edge_feature_dim(self) -> int:
        return self._edge_feature.shape[1]


@dataclass
class DataBatch:
    src_ids: torch.Tensor
    dst_ids: torch.Tensor
    timestamps: torch.Tensor
    edge_ids: torch.Tensor
    labels: torch.Tensor

    @property
    def size(self) -> int:
        return len(self.src_ids)

    def to(self, device: torch.device) -> None:
        self.src_ids.to(device)
        self.dst_ids.to(device)
        self.timestamps.to(device)
        self.edge_ids.to(device)
        self.labels.to(device)


@dataclass
class Dataset:
    name: str
    src_ids: np.ndarray     # int, 1D
    dst_ids: np.ndarray     # int, 1D
    timestamps: np.ndarray  # float, 1D
    edge_ids: np.ndarray  # int, 1D
    labels: np.ndarray      # int, 1D (0 or 1)

    def __post_init__(self) -> None:
        self.n_sample = self.src_ids.shape[0]
        self.unique_nodes = set(self.src_ids) | set(self.dst_ids)

        self._shape_check()

    def _shape_check(self) -> None:
        """Check whether all attributes have correct shapes.
        """
        assert self.src_ids.shape == (self.n_sample,)
        assert self.dst_ids.shape == (self.n_sample,)
        assert self.timestamps.shape == (self.n_sample,)
        assert self.edge_ids.shape == (self.n_sample,)
        assert self.labels.shape == (self.n_sample,)

    @property
    def num_unique_nodes(self) -> int:
        return len(self.unique_nodes)

    def _get_subset_by_indicator(self, name: str, select: np.ndarray) -> Dataset:
        return Dataset(
            name,
            self.src_ids[select], self.dst_ids[select], self.timestamps[select],
            self.labels[select], self.edge_ids[select],
        )

    def get_subset_by_time_range(self, name: str, lower: float = None, upper: float = None) -> Dataset:
        assert lower is not None or upper is not None
        lower = float("-inf") if lower is None else lower
        upper = float("inf") if upper is None else upper
        select: np.ndarray = np.logical_and(lower <= self.timestamps, self.timestamps < upper)
        return self._get_subset_by_indicator(name, select)

    def get_subset_by_removing_nodes(self, name: str, nodes: Set[int]) -> Dataset:
        remove_nodes = np.array(list(nodes))
        remove: np.ndarray = np.logical_or(np.isin(self.src_ids, remove_nodes), np.isin(self.dst_ids, remove_nodes))
        return self._get_subset_by_indicator(name, ~remove)

    def get_subset_by_selecting_nodes(self, name: str, nodes: Set[int]) -> Dataset:
        select_nodes = np.array(list(nodes))
        select: np.ndarray = np.logical_or(np.isin(self.src_ids, select_nodes), np.isin(self.dst_ids, select_nodes))
        return self._get_subset_by_indicator(name, select)

    def describe(self, logger: Logger) -> None:
        logger.info(f"Dataset {self.name} has {self.n_sample} edges and {self.num_unique_nodes} unique nodes.")

    def get_batch_num(self, batch_size: int) -> int:
        return int(math.ceil(self.n_sample / batch_size))

    def batch_generator(self, batch_size: int, device: torch.device) -> Generator[DataBatch, None, None]:
        start_idx = 0
        while start_idx < self.n_sample:
            end_idx = min(self.n_sample, start_idx + batch_size)
            batch = DataBatch(
                src_ids=torch.from_numpy(self.src_ids[start_idx:end_idx]).long().to(device),
                dst_ids=torch.from_numpy(self.dst_ids[start_idx:end_idx]).long().to(device),
                timestamps=torch.from_numpy(self.timestamps[start_idx:end_idx]).float().to(device),
                edge_ids=torch.from_numpy(self.edge_ids[start_idx:end_idx]).long().to(device),
                labels=torch.from_numpy(self.edge_ids[start_idx:end_idx]).long().to(device),
            )
            yield batch
            start_idx = end_idx


def get_self_supervised_data(
    logger: Logger,
    workspace_path: str,
    different_new_nodes_between_val_and_test: bool = False,
    randomize_features: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset, AbsFeatureRepo]:
    # Load data from files
    graph_df = pd.read_csv(f'{workspace_path}/data/graph.csv')
    edge_feature = np.load(f'{workspace_path}/data/edge_feature.npy')
    node_feature = np.load(f'{workspace_path}/data/node_feature.npy')
    if randomize_features:
        node_feature = np.random.rand(*node_feature.shape)

    # Create full data
    full_data = Dataset(
        name='full_data',
        src_ids=graph_df.u.values, dst_ids=graph_df.i.values, timestamps=graph_df.ts.values,
        labels=graph_df.label.values, edge_ids=graph_df.idx.values,
    )
    feature_repo = StaticFeatureRepo(node_feature=node_feature, edge_feature=edge_feature)

    # Create validation data & testing data
    val_time, test_time = list(np.quantile(full_data.timestamps, [0.70, 0.85]))
    val_data = full_data.get_subset_by_time_range('val_data', lower=val_time, upper=test_time)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time)

    # Sample new nodes
    random.seed(2022)
    val_and_test_nodes = set(val_data.src_ids) | set(val_data.dst_ids) | set(test_data.src_ids) | set(test_data.dst_ids)
    new_val_and_test_nodes = set(random.sample(val_and_test_nodes, int(0.1 * full_data.num_unique_nodes)))

    # Create training data
    train_data = full_data.get_subset_by_removing_nodes('', new_val_and_test_nodes)\
        .get_subset_by_time_range('train_data', upper=val_time)
    assert len(train_data.unique_nodes & new_val_and_test_nodes) == 0

    # Create pure new node data
    if different_new_nodes_between_val_and_test:
        node_list = list(new_val_and_test_nodes)
        n = len(node_list) // 2
        val_new_nodes = set(node_list[:n])
        test_new_nodes = set(node_list[n:])
    else:
        val_new_nodes = test_new_nodes = new_val_and_test_nodes
    new_node_val_data = val_data.get_subset_by_selecting_nodes('new_node_val_data', val_new_nodes)
    new_node_test_data = test_data.get_subset_by_selecting_nodes('new_node_test_data', test_new_nodes)

    # Show descriptions of all datasets
    logger.info('===== Data statistics =====')
    for data in [full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data]:
        data.describe(logger)
    logger.info(f'Node feature shape: ({feature_repo.num_nodes()}, {feature_repo.node_feature_dim()})')
    logger.info(f'Edge feature shape: ({feature_repo.num_edges()}, {feature_repo.edge_feature_dim()})')
    logger.info('')

    return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, feature_repo


def get_self_supervised_data_backup(
    logger: Logger,
    workspace_path: str,
    different_new_nodes_between_val_and_test: bool = False,
    randomize_features: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Dataset, Dataset, AbsFeatureRepo]:
    # Load data from files
    graph_df = pd.read_csv(f'{workspace_path}/data/graph.csv')
    edge_feature = np.load(f'{workspace_path}/data/edge_feature.npy')
    node_feature = np.load(f'{workspace_path}/data/node_feature.npy')
    if randomize_features:
        node_feature = np.random.rand(*node_feature.shape)

    # Create full data
    full_data = Dataset(
        name='full_data',
        src_ids=graph_df.u.values, dst_ids=graph_df.i.values, timestamps=graph_df.ts.values,
        labels=graph_df.label.values, edge_ids=graph_df.idx.values,
    )
    feature_repo = StaticFeatureRepo(node_feature=node_feature, edge_feature=edge_feature)

    # Create validation data & testing data
    val_time, test_time = list(np.quantile(full_data.timestamps, [0.70, 0.85]))
    val_data = full_data.get_subset_by_time_range('val_data', lower=val_time + 1e-5, upper=test_time + 1e-5)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time + 1e-5)

    # Sample new nodes
    random.seed(2020)
    val_and_test_nodes = set(val_data.src_ids) | set(val_data.dst_ids) | set(test_data.src_ids) | set(test_data.dst_ids)
    new_val_and_test_nodes = set(random.sample(sorted(list(val_and_test_nodes)), int(0.1 * full_data.num_unique_nodes)))

    # Create training data
    train_data = full_data.get_subset_by_removing_nodes('', new_val_and_test_nodes)\
        .get_subset_by_time_range('train_data', upper=val_time + 1e-5)
    assert len(train_data.unique_nodes & new_val_and_test_nodes) == 0

    # Create pure new node data
    if different_new_nodes_between_val_and_test:
        node_list = list(new_val_and_test_nodes)
        n = len(node_list) // 2
        val_new_nodes = set(node_list[:n])
        test_new_nodes = set(node_list[n:])
    else:
        # val_new_nodes = test_new_nodes = new_val_and_test_nodes
        val_new_nodes = test_new_nodes = full_data.unique_nodes - train_data.unique_nodes
    new_node_val_data = val_data.get_subset_by_selecting_nodes('new_node_val_data', val_new_nodes)
    new_node_test_data = test_data.get_subset_by_selecting_nodes('new_node_test_data', test_new_nodes)

    # Show descriptions of all datasets
    logger.info('===== Data statistics =====')
    for data in [full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data]:
        data.describe(logger)
    logger.info(f'Node feature shape: ({feature_repo.num_nodes()}, {feature_repo.node_feature_dim()})')
    logger.info(f'Edge feature shape: ({feature_repo.num_edges()}, {feature_repo.edge_feature_dim()})')
    logger.info('')

    return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, feature_repo
