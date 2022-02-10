import argparse
import os
import pathlib
import shutil
import time

import numpy as np
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from data import Dataset, get_neighbor_finder, get_self_supervised_data
from evaluation import evaluate_edge_prediction
from model import AbsEmbeddingModel
from utils import (
    EarlyStopMonitor, get_module, load_model, make_logger, RandomNodeSelector, save_model,
    WorkflowContext
)
from utils.training import copy_best_model


def train_single_epoch(
    workflow_context: WorkflowContext,
    emb_model: AbsEmbeddingModel, data: Dataset, batch_size: int,
    backprop_every: int, device: torch.device, optimizer: Optimizer,
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
    emb_model.epoch_start_step()
    emb_model.train()
    for pos_batch in tqdm(batch_generator, total=batch_num, desc=f'Training progress', unit='batch'):
        # Forward propagation
        batch_size = pos_batch.size
        pos_batch.neg_ids = torch.from_numpy(random_node_selector.sample(batch_size)).long()
        pos_prob, neg_prob = emb_model.compute_edge_probabilities(pos_batch)
        with torch.no_grad():
            pos_label = torch.ones(batch_size, dtype=torch.float)
            neg_label = torch.zeros(batch_size, dtype=torch.float)
        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        if iter_cnt % backprop_every == 0 or sample_cnt == data.n_sample:  # Back propagation
            assert isinstance(loss, torch.Tensor)
            loss /= backprop_every
            loss.backward()
            optimizer.step()
            loss_records.append(loss.item())

            loss = 0
            optimizer.zero_grad()
            emb_model.backward_post_step()

        sample_cnt += pos_batch.size
        iter_cnt += 1
        if workflow_context.dry_run and iter_cnt >= workflow_context.dry_run_iter_limit:
            break
    emb_model.epoch_end_step()

    return float(np.mean(loss_records))


def prepare_workspace(version_path: str) -> None:
    if os.path.exists(version_path):
        raise ValueError(f'{version_path} already exists.')

    os.mkdir(version_path)
    os.mkdir(os.path.join(version_path, 'saved_models'))

    # Backup code
    current_path = os.path.abspath(pathlib.Path())
    backup_path = os.path.join(version_path, '_code_backup')
    shutil.copytree(current_path, backup_path)


def run_train_self_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']
    version_path = f'{workspace_path}/version__{args.version}/'
    prepare_workspace(version_path)

    workflow_context = WorkflowContext(
        logger=make_logger(os.path.join(version_path, 'log.txt')),
        dry_run=args.dry,
    )

    # Run model test
    if workflow_context.dry_run:
        workflow_context.logger.info('Run workflow in dry mode.')

    # Read data
    data_dict, feature_repo = \
        get_self_supervised_data(
            workflow_context,
            workspace_path=workspace_path,
            randomize_features=config['data']['randomize_features']
        )

    # Neighbor finder
    full_neighbor_finder = get_neighbor_finder(data_dict['full'], uniform=config['data']['uniform'])
    train_neighbor_finder = get_neighbor_finder(data_dict['train'], uniform=config['data']['uniform'])

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
    get_emb_model = getattr(job_module, 'get_emb_model')
    emb_model = get_emb_model(feature_repo, device)
    assert isinstance(emb_model, AbsEmbeddingModel)

    num_epoch = train_config['num_epoch']
    early_stopper = EarlyStopMonitor(max_round=train_config['early_stop'])
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=lr)
    for epoch in range(1, num_epoch + 1):
        workflow_context.logger.info(f'===== Start epoch {epoch} =====')
        workflow_context.logger.info(
            f'Run backprop every {backprop_every} {"iteration" if backprop_every == 1 else "iterations"}.'
        )
        epoch_start_time = time.time()

        emb_model.set_neighbor_finder(train_neighbor_finder)
        train_loss = train_single_epoch(
            workflow_context,
            emb_model, data_dict['train'], train_batch_size, backprop_every, device, optimizer,
        )
        workflow_context.logger.info(f'Training finished. Mean training loss = {train_loss:.6f}')

        save_model(workflow_context, model=emb_model, version_path=version_path, epoch=epoch)

        emb_model.set_neighbor_finder(full_neighbor_finder)
        ap, auc = evaluate_edge_prediction(
            workflow_context,
            emb_model, data_dict['eval'], eval_batch_size, evaluation_seed, device,
        )
        workflow_context.logger.info(f'Evaluation result: AP = {ap:.6f}, AUC = {auc:.6f}.')

        if early_stopper.early_stop_check(ap):
            workflow_context.logger.info(f'No improvement over {train_config["early_stop"]} rounds, early stop.')
            workflow_context.logger.info(f'Best epoch = {early_stopper.best_epoch}')
            break

        epoch_end_time = time.time()
        workflow_context.logger.info(f'=== Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.')
        workflow_context.logger.info('')

    copy_best_model(version_path=version_path, best_epoch=early_stopper.best_epoch)

    load_model(workflow_context, emb_model, version_path=version_path)
    emb_model.set_neighbor_finder(full_neighbor_finder)
    ap, auc = evaluate_edge_prediction(
        workflow_context,
        emb_model, data_dict['eval'], eval_batch_size, evaluation_seed, device,
    )
    workflow_context.logger.info(f'Reload model and test. Evaluation result: AP = {ap:.6f}, AUC = {auc:.6f}.')
