import os
from logging import Logger

import numpy as np
import pandas as pd

from data import Dataset


def process_data(
    logger: Logger,
    workspace_path: str,
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
    train_data = full_data.get_subset_by_time_range('train_data', upper=eval_time + 1e-5)
    eval_data = full_data.get_subset_by_time_range('eval_data', lower=eval_time + 1e-5, upper=test_time + 1e-5)
    test_data = full_data.get_subset_by_time_range('test_data', lower=test_time + 1e-5)

    for data in [full_data, train_data, eval_data, test_data]:
        data.describe(logger)

    full_data.to_csv(os.path.join(workspace_path, 'data', 'graph_supervised_full.csv'))
    train_data.to_csv(os.path.join(workspace_path, 'data', 'graph_supervised_train.csv'))
    eval_data.to_csv(os.path.join(workspace_path, 'data', 'graph_supervised_eval.csv'))
    test_data.to_csv(os.path.join(workspace_path, 'data', 'graph_supervised_test.csv'))
