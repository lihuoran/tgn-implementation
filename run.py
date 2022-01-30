import argparse
import os
import yaml

from workflow.train_self_supervised import run_train_self_supervised


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('TGN workflow')
    parser.add_argument('--job_path', type=str, help='Job folder path', required=True)
    parser.add_argument('--version', type=str, help='Job folder path', required=True)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    config_path = os.path.join(os.path.abspath(args.job_path), 'config.yml')
    config = yaml.safe_load(open(config_path))

    if config['job_type'] == 'train_self_supervised':
        run_train_self_supervised(args, config)


if __name__ == '__main__':
    main(get_args())
