import argparse
import os

from utils import get_module, make_logger, WorkflowContext


def run_data_self_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']

    workflow_context = WorkflowContext(
        logger=make_logger(os.path.join(workspace_path, 'data', 'log_self_supervised.txt'))
    )

    job_module = get_module(os.path.abspath(args.job_path))
    process_data = getattr(job_module, 'process_data')
    process_data(
        workflow_context,
        workspace_path,
        different_new_nodes_between_val_and_test=config['different_new_nodes'],
    )
