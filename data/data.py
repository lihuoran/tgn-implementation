from __future__ import annotations

import random
from typing import List, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    src_id: np.ndarray     # int, 1D
    dst_id: np.ndarray     # int, 1D
    ts: np.ndarray         # float, 1D
    label: np.ndarray      # int, 1D (0 or 1)
    edge_idx: np.ndarray   # int, 1D
    edge_feat: np.ndarray  # float, 2D
    node_feat: np.ndarray  # float, 2D

    def _shape_check(self) -> None:
        """Check whether all attributes have correct shapes.
        """
        assert self.src_id.shape == (self.n_sample,)
        assert self.dst_id.shape == (self.n_sample,)
        assert self.ts.shape == (self.n_sample,)
        assert self.label.shape == (self.n_sample,)
        assert self.edge_idx.shape == (self.n_sample,)
        assert len(self.edge_feat.shape) == 2
        assert len(self.node_feat.shape) == 2

    def __post_init__(self) -> None:
        self.n_sample = self.src_id.shape[0]
        self.unique_nodes = set(self.src_id) | set(self.dst_id)

        self._shape_check()

    @property
    def n_unique_nodes(self) -> int:
        return len(self.unique_nodes)

    def _get_subset_by_indicator(self, name: str, select: np.ndarray) -> Dataset:
        return Dataset(
            name,
            self.src_id[select], self.dst_id[select], self.ts[select],
            self.label[select], self.edge_idx[select],
            # Do not duplicate feature matrices
            self.edge_feat, self.node_feat,
        )

    def get_subset_by_time_range(self, name: str, lower: float = None, upper: float = None) -> Dataset:
        assert lower is not None or upper is not None
        lower = float("-inf") if lower is None else lower
        upper = float("inf") if upper is None else upper
        select: np.ndarray = lower <= self.ts & self.ts < upper
        return self._get_subset_by_indicator(name, select)

    def get_subset_by_removing_nodes(self, name: str, nodes: Set[int]) -> Dataset:
        remove_nodes = np.array(list(nodes))
        remove: np.ndarray = np.isin(self.src_id, remove_nodes) | np.isin(self.dst_id, remove_nodes)
        return self._get_subset_by_indicator(name, ~remove)

    def get_subset_by_selecting_nodes(self, name: str, nodes: Set[int]) -> Dataset:
        remove_nodes = np.array(list(nodes))
        select: np.ndarray = np.isin(self.src_id, remove_nodes) | np.isin(self.dst_id, remove_nodes)
        return self._get_subset_by_indicator(name, select)

    def describe(self) -> None:
        print(f"Dataset {self.name} has {self.n_sample} edges and {self.n_unique_nodes} unique nodes.")


def get_self_supervised_data(
    workspace_path: str,
    dataset_name: str,
    different_new_nodes_between_val_and_test: bool = False,
    randomize_features: bool = False,
) -> List[Dataset]:
    # Load data from files
    graph_df = pd.read_csv(f'{workspace_path}/data/ml_{dataset_name}.csv')
    edge_feat = np.load(f'{workspace_path}/data/ml_{dataset_name}.npy')
    node_feat = np.load(f'{workspace_path}/data/ml_{dataset_name}_node.npy')
    if randomize_features:
        node_feat = np.random.rand(*node_feat.shape)

    # Create full data
    full_data = Dataset(
        name='full_data',
        src_id=graph_df.u.values, dst_id=graph_df.i.values, ts=graph_df.ts.values,
        label=graph_df.label.values, edge_idx=graph_df.idx.values,
        edge_feat=edge_feat, node_feat=node_feat,
    )

    # Create validation data & testing data
    val_time, test_time = list(np.quantile(full_data.ts, [0.70, 0.85]))
    val_data = full_data.get_subset_by_time_range('val_data', lower=val_time, upper=test_time)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time)

    # Sample new nodes
    random.seed(2022)
    val_and_test_nodes = set(val_data.src_id) | set(val_data.dst_id) | set(test_data.src_id) | set(test_data.dst_id)
    new_val_and_test_nodes = set(random.sample(val_and_test_nodes, int(0.1 * full_data.n_unique_nodes)))

    # Create training data
    train_data = full_data.get_subset_by_removing_nodes('train_data', new_val_and_test_nodes)
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
    datasets = [full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data]
    for data in datasets:
        data.describe()

    return datasets


if __name__ == '__main__':
    get_self_supervised_data('C:/workspace/tgn-workspace', 'wikipedia', True)
