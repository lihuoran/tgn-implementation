import argparse
import os
import pathlib
import shutil
import time

import numpy as np
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from data import Dataset, get_data
from evaluation import evaluate_node_classification
from model import AbsBinaryClassificationModel, AbsEmbeddingModel
from utils import (
    EarlyStopMonitor, get_module, get_neighbor_finder, load_model, make_logger, save_model
)
from utils.training import copy_best_model


def train_single_epoch(
    emb_model: AbsEmbeddingModel, cls_model: AbsBinaryClassificationModel, data: Dataset, batch_size: int,
    backprop_every: int, device: torch.device, optimizer: Optimizer,
    dry_run: bool = False
) -> float:
    criterion = torch.nn.BCELoss()

    iter_cnt = 0
    sample_cnt = 0
    loss = 0
    optimizer.zero_grad()
    loss_records = []

    batch_num = data.get_batch_num(batch_size)
    batch_generator = data.batch_generator(batch_size, device)

    emb_model.epoch_start_step()
    emb_model.train()
    for batch in tqdm(batch_generator, total=batch_num, desc=f'Training progress', unit='batch'):
        # Forward propagation
        src_emb, dst_emb, _ = emb_model.compute_temporal_embeddings(batch)
        src_prob = cls_model.get_prob(src_emb)
        src_pred = src_prob.sigmoid()

        loss += criterion(src_pred, batch.labels)

        if iter_cnt % backprop_every == 0 or sample_cnt == data.n_sample:  # Back propagation
            assert isinstance(loss, torch.Tensor)
            loss /= backprop_every
            loss.backward()
            optimizer.step()
            loss_records.append(loss.item())

            loss = 0
            optimizer.zero_grad()
            emb_model.backward_post_step()

        sample_cnt += batch.size
        iter_cnt += 1
        if dry_run and iter_cnt == 5:
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


def run_train_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']
    version_path = f'{workspace_path}/version__{args.version}/'
    prepare_workspace(version_path)
    logger = make_logger(os.path.join(version_path, 'log.txt'))

    # Read data
    data_dict, feature_repo = \
        get_data(
            logger=logger,
            workspace_path=workspace_path,
            require_new_node_data=False,
            randomize_features=config['data']['randomize_features']
        )

    # Neighbor finder
    train_neighbor_finder = get_neighbor_finder(data_dict['train'], uniform=config['data']['uniform'])

    # Training config
    train_config = config['training']
    device_name: str = train_config['device']
    device: torch.device = torch.device(device_name) if device_name is not None \
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = train_config['train_batch_size']
    backprop_every = train_config['backprop_every']
    lr = train_config['learning_rate']

    embedding_model_version = train_config['embedding_model_version']
    embedding_model_version_path = f'{workspace_path}/version__{embedding_model_version}/'

    # Evaluation config
    eval_batch_size = train_config.get('eval_batch_size', train_batch_size)
    evaluation_seed = train_config['evaluation_seed']

    # Job module & objects
    job_module = get_module(os.path.abspath(args.job_path))
    #
    get_emb_model = getattr(job_module, 'get_emb_model')
    emb_model = get_emb_model(feature_repo, device)
    assert isinstance(emb_model, AbsEmbeddingModel)
    load_model(emb_model, version_path=embedding_model_version_path)  # Load pretrained embedding model
    emb_model.set_neighbor_finder(train_neighbor_finder)
    emb_model.eval()   # Freeze embedding model. Do not train this.
    #
    get_cls_model = getattr(job_module, 'get_cls_model')
    cls_model = get_cls_model(feature_repo.node_feature_dim(), device)
    assert isinstance(cls_model, AbsBinaryClassificationModel)

    num_epoch = train_config['num_epoch']
    early_stopper = EarlyStopMonitor(max_round=train_config['early_stop'])
    optimizer = torch.optim.Adam(cls_model.parameters(), lr=lr)
    for epoch in range(1, num_epoch + 1):
        logger.info(f'===== Start epoch {epoch} =====')
        logger.info(f'Run backprop every {backprop_every} {"iteration" if backprop_every == 1 else "iterations"}.')
        epoch_start_time = time.time()

        emb_model.set_neighbor_finder(train_neighbor_finder)
        train_loss = train_single_epoch(
            emb_model, cls_model, data_dict['train'], train_batch_size, backprop_every, device, optimizer,
            dry_run=args.dry
        )
        logger.info(f'Training finished. Mean training loss = {train_loss:.6f}')

        save_model(model=cls_model, version_path=version_path, epoch=epoch)

        auc = evaluate_node_classification(
            emb_model, cls_model, data_dict['eval'], eval_batch_size, evaluation_seed, device, dry_run=args.dry
        )
        logger.info(f'Evaluation result: AUC = {auc:.6f}.')

        if early_stopper.early_stop_check(auc):
            logger.info(f'No improvement over {train_config["early_stop"]} rounds, early stop.')
            logger.info(f'Best epoch = {early_stopper.best_epoch}')
            break

        epoch_end_time = time.time()
        logger.info(f'=== Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.')
        logger.info('')

    copy_best_model(version_path=version_path, best_epoch=early_stopper.best_epoch)

    load_model(emb_model, version_path=version_path)
    auc = evaluate_node_classification(
        emb_model, cls_model, data_dict['eval'], eval_batch_size, evaluation_seed, device, dry_run=args.dry
    )
    logger.info(f'Reload model and test. Evaluation result: AUC = {auc:.6f}.')
