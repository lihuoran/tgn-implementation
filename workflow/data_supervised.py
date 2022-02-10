import argparse
import os

from utils import get_module, make_logger, WorkflowContext


def run_data_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']

    workflow_context = WorkflowContext(
        logger=make_logger(os.path.join(workspace_path, 'data', 'log.txt'))
    )

    job_module = get_module(os.path.abspath(args.job_path))
    process_data = getattr(job_module, 'process_data')
    process_data(workflow_context, workspace_path)
