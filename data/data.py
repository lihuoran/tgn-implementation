from __future__ import annotations

import math
import os
import random
from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Dict, Generator, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import WorkflowContext


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
    neg_ids: torch.Tensor = None

    @property
    def size(self) -> int:
        return len(self.src_ids)

    def to(self, device: torch.device) -> None:
        self.src_ids.to(device)
        self.dst_ids.to(device)
        self.timestamps.to(device)
        self.edge_ids.to(device)
        self.labels.to(device)
        if self.neg_ids is not None:
            self.neg_ids.to(device)


@dataclass
class Dataset:
    name: str
    src_ids: np.ndarray  # int, 1D
    dst_ids: np.ndarray  # int, 1D
    timestamps: np.ndarray  # float, 1D
    edge_ids: np.ndarray  # int, 1D
    labels: np.ndarray  # float, 1D

    def __post_init__(self) -> None:
        self.n_sample = self.src_ids.shape[0]
        self.unique_nodes = set(self.src_ids) | set(self.dst_ids)

        self._shape_check()
        self._time_order_check()

    def show_debug_info(self) -> None:
        print(self.n_sample)
        print(self.src_ids[:10])
        print(self.dst_ids[:10])
        print(self.timestamps[:10])
        print(self.edge_ids[:10])
        print(self.labels[:10])
        print(self.src_ids[-10:])
        print(self.dst_ids[-10:])
        print(self.timestamps[-10:])
        print(self.edge_ids[-10:])
        print(self.labels[-10:])

    def _shape_check(self) -> None:
        """Check whether all attributes have correct shapes.
        """
        assert self.src_ids.shape == (self.n_sample,)
        assert self.dst_ids.shape == (self.n_sample,)
        assert self.timestamps.shape == (self.n_sample,)
        assert self.edge_ids.shape == (self.n_sample,)
        assert self.labels.shape == (self.n_sample,)
        print(f'{self.name} shape check passed.')

    def _time_order_check(self) -> None:
        n = len(self.timestamps)
        for i in range(n - 1):
            assert self.timestamps[i] <= self.timestamps[i + 1]
        print(f'{self.name} time order check passed.')

    @property
    def num_unique_nodes(self) -> int:
        return len(self.unique_nodes)

    def _get_subset_by_indicator(self, name: str, select: np.ndarray) -> Dataset:
        return Dataset(
            name,
            src_ids=self.src_ids[select],
            dst_ids=self.dst_ids[select],
            timestamps=self.timestamps[select],
            edge_ids=self.edge_ids[select],
            labels=self.labels[select],
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

    def describe(self, workflow_context: WorkflowContext) -> None:
        msg = f"Dataset {self.name} has {self.n_sample} edges and {self.num_unique_nodes} unique nodes."
        workflow_context.logger.info(msg)

    def get_batch_num(self, batch_size: int) -> int:
        return int(math.ceil(self.n_sample / batch_size))

    def batch_generator(
        self,
        workflow_context: WorkflowContext,
        batch_size: int,
        device: torch.device,
        desc: str = None,
    ) -> Generator[DataBatch, None, None]:
        batch_num = int(math.ceil(self.n_sample / batch_size))
        if workflow_context.dry_run:
            batch_num = min(batch_num, workflow_context.dry_run_iter_limit)

        start_idx = 0
        for _ in tqdm(range(batch_num), desc=desc, unit='batch'):
            end_idx = min(self.n_sample, start_idx + batch_size)
            batch = DataBatch(
                src_ids=torch.from_numpy(self.src_ids[start_idx:end_idx]).long().to(device),
                dst_ids=torch.from_numpy(self.dst_ids[start_idx:end_idx]).long().to(device),
                timestamps=torch.from_numpy(self.timestamps[start_idx:end_idx]).float().to(device),
                edge_ids=torch.from_numpy(self.edge_ids[start_idx:end_idx]).long().to(device),
                labels=torch.from_numpy(self.labels[start_idx:end_idx]).float().to(device),
            )
            yield batch

            start_idx = end_idx

    @staticmethod
    def from_csv(name: str, path: str, nrows: int = None) -> Dataset:
        graph_df = pd.read_csv(path, nrows=nrows)
        return Dataset(
            name=name,
            src_ids=graph_df.u.values,
            dst_ids=graph_df.i.values,
            timestamps=graph_df.ts.values,
            labels=graph_df.label.values,
            edge_ids=graph_df.idx.values,
        )

    def to_csv(self, path: str) -> None:
        df = pd.DataFrame({
            'u': self.src_ids,
            'i': self.dst_ids,
            'ts': self.timestamps,
            'label': self.labels,
            'idx': self.edge_ids,
        })
        df.to_csv(path)


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
    val_data = full_data.get_subset_by_time_range('val_data', lower=val_time, upper=test_time)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time)

    # Sample new nodes
    random.seed(2022)
    val_and_test_nodes = set(val_data.src_ids) | set(val_data.dst_ids) | set(test_data.src_ids) | set(test_data.dst_ids)
    new_val_and_test_nodes = set(random.sample(val_and_test_nodes, int(0.1 * full_data.num_unique_nodes)))

    # Create training data
    train_data = full_data.get_subset_by_removing_nodes('', new_val_and_test_nodes) \
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

    assert train_data.timestamps[-1] < val_data.timestamps[0]
    assert val_data.timestamps[-1] < test_data.timestamps[0]

    return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, feature_repo


def get_self_supervised_data(
    workflow_context: WorkflowContext,
    workspace_path: str,
    randomize_features: bool = False,
) -> Tuple[Dict[str, Dataset], AbsFeatureRepo]:
    edge_feature = np.load(f'{workspace_path}/data/edge_feature.npy')
    node_feature = np.load(f'{workspace_path}/data/node_feature.npy')
    if randomize_features:
        node_feature = np.random.rand(*node_feature.shape)
    feature_repo = StaticFeatureRepo(node_feature=node_feature, edge_feature=edge_feature)

    path_template = os.path.join(workspace_path, 'data', 'graph_self_supervised_{}.csv')
    data_keys = ['full', 'train', 'eval', 'test', 'nn_eval', 'nn_test']
    data_dict: Dict[str, Dataset] = {
        name: Dataset.from_csv(name=name, path=path_template.format(name))
        for name in data_keys
    }

    # Show descriptions of all datasets
    workflow_context.logger.info('===== Data statistics =====')
    for data in data_dict.values():
        data.describe(workflow_context)
    workflow_context.logger.info(f'Node feature shape: ({feature_repo.num_nodes()}, {feature_repo.node_feature_dim()})')
    workflow_context.logger.info(f'Edge feature shape: ({feature_repo.num_edges()}, {feature_repo.edge_feature_dim()})')
    workflow_context.logger.info('')

    return data_dict, feature_repo


def get_supervised_data(
    workflow_context: WorkflowContext,
    workspace_path: str,
    randomize_features: bool = False,
) -> Tuple[Dict[str, Dataset], AbsFeatureRepo]:
    edge_feature = np.load(f'{workspace_path}/data/edge_feature.npy')
    node_feature = np.load(f'{workspace_path}/data/node_feature.npy')
    if randomize_features:
        node_feature = np.random.rand(*node_feature.shape)
    feature_repo = StaticFeatureRepo(node_feature=node_feature, edge_feature=edge_feature)

    path_template = os.path.join(workspace_path, 'data', 'graph_supervised_{}.csv')
    data_keys = ['full', 'train', 'eval', 'test']
    data_dict: Dict[str, Dataset] = {
        name: Dataset.from_csv(name=name, path=path_template.format(name))
        for name in data_keys
    }

    # Show descriptions of all datasets
    workflow_context.logger.info('===== Data statistics =====')
    for data in data_dict.values():
        data.describe(workflow_context)
    workflow_context.logger.info(f'Node feature shape: ({feature_repo.num_nodes()}, {feature_repo.node_feature_dim()})')
    workflow_context.logger.info(f'Edge feature shape: ({feature_repo.num_edges()}, {feature_repo.edge_feature_dim()})')
    workflow_context.logger.info('')

    return data_dict, feature_repo
