import argparse
import os

from utils.log import make_logger
from utils.path import get_module


def run_data_self_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']
    logger = make_logger(os.path.join(workspace_path, 'data', 'log.txt'))

    job_module = get_module(os.path.abspath(args.job_path))
    process_data = getattr(job_module, 'process_data')
    process_data(
        logger,
        workspace_path,
        different_new_nodes_between_val_and_test=config['different_new_nodes'],
    )
