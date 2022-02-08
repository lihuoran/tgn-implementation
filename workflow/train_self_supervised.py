import argparse
import dataclasses
import os
import time

import numpy as np
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from data.data import Dataset, get_self_supervised_data
from evaluation.evaluation import evaluate_edge_prediction
from model.abs_model import AbsModel
from utils.log import make_logger
from utils.path import get_module
from utils.training import EarlyStopMonitor, get_neighbor_finder, RandomNodeSelector, save_model


def train_single_epoch(
    model: AbsModel, data: Dataset, batch_size: int, backprop_every: int, device: torch.device, optimizer: Optimizer
) -> float:
    criterion = torch.nn.BCELoss()

    iter_cnt = 0
    sample_cnt = 0
    loss = 0
    optimizer.zero_grad()
    loss_records = []

    batch_num = data.get_batch_num(batch_size)
    batch_generator = data.batch_generator(batch_size, device)

    random_node_selector = RandomNodeSelector(data.dst_ids)
    model.epoch_start_step()
    model.train_mode()
    for pos_batch in tqdm(batch_generator, total=batch_num, desc=f'Training progress', unit='batch'):
        # Forward propagation
        neg_batch = dataclasses.replace(pos_batch)
        neg_batch.dst_ids = torch.from_numpy(random_node_selector.sample(neg_batch.size))

        model.train_mode()
        pos_prob = model.compute_edge_probabilities(pos_batch)
        neg_prob = model.compute_edge_probabilities(neg_batch)
        with torch.no_grad():
            pos_label = torch.ones(pos_batch.size, dtype=torch.float)
            neg_label = torch.zeros(neg_batch.size, dtype=torch.float)
        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        sample_cnt += pos_batch.size
        iter_cnt += 1
        if iter_cnt % backprop_every == 0 or sample_cnt == data.n_sample:  # Back propagation
            assert isinstance(loss, torch.Tensor)
            loss /= backprop_every
            loss.backward()
            optimizer.step()
            loss_records.append(loss.item())

            loss = 0
            optimizer.zero_grad()

            model.backward_post_step()
    model.epoch_end_step()

    return float(np.mean(loss_records))


def prepare_workspace(version_path: str) -> None:
    if os.path.exists(version_path):
        raise ValueError(f'{version_path} already exists.')

    os.mkdir(version_path)
    os.mkdir(os.path.join(version_path, 'saved_models'))

    # Backup code
    # current_path = os.path.abspath(pathlib.Path())
    # backup_path = os.path.join(version_path, '_code_backup')
    # shutil.copytree(current_path, backup_path)


def run_train_self_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']
    version_path = f'{workspace_path}/version__{args.version}/'
    prepare_workspace(version_path)
    logger = make_logger(os.path.join(version_path, 'log.txt'))

    # Read data
    full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, feature_repo = \
        get_self_supervised_data(
            logger=logger,
            workspace_path=workspace_path,
            different_new_nodes_between_val_and_test=config['data']['different_new_nodes'],
            randomize_features=config['data']['randomize_features']
        )

    # Neighbor finder
    full_neighbor_finder = get_neighbor_finder(full_data, uniform=config['data']['uniform'])
    train_neighbor_finder = get_neighbor_finder(train_data, uniform=config['data']['uniform'])

    # Training config
    train_config = config['training']
    device_name: str = train_config['device']
    device: torch.device = torch.device(device_name) if device_name is not None \
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = train_config['train_batch_size']
    backprop_every = train_config['backprop_every']
    lr = train_config['learning_rate']

    # Evaluation config
    eval_batch_size = train_config.get('eval_batch_size', train_batch_size)
    evaluation_seed = train_config['evaluation_seed']

    # Job module & objects
    job_module = get_module(os.path.abspath(args.job_path))
    get_model = getattr(job_module, 'get_model')
    model = get_model(feature_repo, device)
    assert isinstance(model, AbsModel)

    num_epoch = train_config['num_epoch']
    early_stopper = EarlyStopMonitor(max_round=train_config['early_stop'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epoch + 1):
        logger.info(f'===== Start epoch {epoch} =====')
        logger.info(f'Run backprop every {backprop_every} {"iteration" if backprop_every == 1 else "iterations"}.')
        epoch_start_time = time.time()

        model.set_neighbor_finder(train_neighbor_finder)
        train_loss = train_single_epoch(model, train_data, train_batch_size, backprop_every, device, optimizer)
        logger.info(f'Training finished. Mean training loss = {train_loss:.6f}')

        model.set_neighbor_finder(full_neighbor_finder)
        ap, auc = evaluate_edge_prediction(model, val_data, eval_batch_size, evaluation_seed, device)
        logger.info(f'Evaluation result: AP = {ap:.6f}, AUC = {auc:.6f}.')

        if early_stopper.early_stop_check(ap):
            logger.info(f'No improvement over {train_config["early_stop"]} rounds, early stop.')
            break

        save_model(version_path=version_path, epoch=epoch, model=model)

        epoch_end_time = time.time()
        logger.info(f'=== Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.')
        logger.info('')

    # eval_model = load_model(version_path=version_path, epoch=num_epoch)
    # assert isinstance(eval_model, AbsModel)
    # ap, auc = evaluate_edge_prediction(eval_model, val_data, eval_batch_size, seed=evaluation_seed)
    # logger.info(f'Reload model and test. Evaluation result: AP = {ap:.6f}, AUC = {auc:.6f}.')
