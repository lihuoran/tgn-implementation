import argparse
import os

import torch.autograd
import yaml

from workflow import run_data_self_supervised, run_data_supervised, run_train_self_supervised, run_train_supervised


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('TGN workflow')
    parser.add_argument('--job_path', type=str, help='Job folder path', required=True)
    parser.add_argument('--version', type=str, help='Version', required=False)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    config_path = os.path.join(os.path.abspath(args.job_path), 'config.yml')
    config = yaml.safe_load(open(config_path))

    if config['job_type'] == 'train_self_supervised':
        run_train_self_supervised(args, config)
    elif config['job_type'] == 'train_supervised':
        run_train_supervised(args, config)
    elif config['job_type'] == 'data_self_supervised':
        run_data_self_supervised(args, config)
    elif config['job_type'] == 'data_supervised':
        run_data_supervised(args, config)
    else:
        raise ValueError(f'Unrecognized job type: {config["job_type"]}')


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)  # TODO: test only
    main(get_args())
