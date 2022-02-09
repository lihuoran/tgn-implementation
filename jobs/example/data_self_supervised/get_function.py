import os
import random
from logging import Logger

import numpy as np
import pandas as pd

from data import Dataset


def process_data(
    logger: Logger,
    workspace_path: str,
    different_new_nodes_between_val_and_test: bool = False,
) -> None:
    # Load data from files
    graph_df = pd.read_csv(os.path.join(workspace_path, 'data', 'graph.csv'))

    # Create full data
    full_data = Dataset(
        name='full_data',
        src_ids=graph_df.u.values, dst_ids=graph_df.i.values, timestamps=graph_df.ts.values,
        labels=graph_df.label.values, edge_ids=graph_df.idx.values,
    )

    # Create validation data & testing data
    eval_time, test_time = list(np.quantile(full_data.timestamps, [0.70, 0.85]))
    eval_data = full_data.get_subset_by_time_range('eval_data', lower=eval_time + 1e-5, upper=test_time + 1e-5)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time + 1e-5)

    # Sample new nodes
    random.seed(2020)
    eval_and_test_nodes = set(eval_data.src_ids) | set(eval_data.dst_ids) | \
        set(test_data.src_ids) | set(test_data.dst_ids)
    new_eval_and_test_nodes = set(
        random.sample(sorted(list(eval_and_test_nodes)), int(0.1 * full_data.num_unique_nodes))
    )

    # Create training data
    train_data = full_data.get_subset_by_removing_nodes('', new_eval_and_test_nodes) \
        .get_subset_by_time_range('train_data', upper=eval_time + 1e-5)
    assert len(train_data.unique_nodes & new_eval_and_test_nodes) == 0

    # Create pure new node data
    if different_new_nodes_between_val_and_test:
        node_list = list(new_eval_and_test_nodes)
        n = len(node_list) // 2
        eval_new_nodes = set(node_list[:n])
        test_new_nodes = set(node_list[n:])
    else:
        eval_new_nodes = test_new_nodes = full_data.unique_nodes - train_data.unique_nodes
    nn_eval_data = eval_data.get_subset_by_selecting_nodes('nn_eval_data', eval_new_nodes)
    nn_test_data = test_data.get_subset_by_selecting_nodes('nn_test_data', test_new_nodes)

    for data in [full_data, train_data, eval_data, test_data, nn_eval_data, nn_test_data]:
        data.describe(logger)

    full_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_full.csv'))
    train_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_train.csv'))
    eval_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_eval.csv'))
    test_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_test.csv'))
    nn_eval_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_nn_eval.csv'))
    nn_test_data.to_csv(os.path.join(workspace_path, 'data', 'graph_self_supervised_nn_test.csv'))
