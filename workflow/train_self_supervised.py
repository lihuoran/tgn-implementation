import argparse
import dataclasses
import math
import os
import time

import torch
from tqdm import tqdm

from data.data import Dataset, get_self_supervised_data
from evaluation.evaluation import evaluate_edge_prediction
from model.tgn import TGN
from utils.log import make_logger
from utils.training import RandomNodeSelector


def train_single_epoch(tgn: TGN, data: Dataset, batch_size: int, backprop_every: int) -> None:
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters())

    iter_cnt = 0
    sample_cnt = 0
    loss = 0
    optimizer.zero_grad()

    batch_num = int(math.ceil(data.n_sample / batch_size))

    random_node_selector = RandomNodeSelector(data.dst_ids)
    tgn.train_mode()
    for pos_batch in tqdm(data.batch_generator(batch_size), total=batch_num, desc=f'Training progress', unit='batch'):
        # Forward propagation
        neg_batch = dataclasses.replace(pos_batch)
        neg_batch.dst_ids = torch.from_numpy(random_node_selector.sample(neg_batch.size))

        tgn.train_mode()
        pos_prob = tgn.compute_edge_probabilities(pos_batch)
        neg_prob = tgn.compute_edge_probabilities(neg_batch)
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
            loss = 0
            optimizer.zero_grad()


def run_train_self_supervised(args: argparse.Namespace, config: dict) -> None:
    # Check & create version folder
    workspace_path = config['workspace_path']
    version_path = f'{workspace_path}/version__{args.version}/'
    if os.path.exists(version_path):
        raise ValueError(f'{version_path} already exists.')
    os.mkdir(version_path)

    # Logger
    logger = make_logger(os.path.join(version_path, 'log.txt'))

    # Read data
    full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, feature_repo = \
        get_self_supervised_data(workspace_path=workspace_path)

    #
    tgn = TGN(
        num_nodes=feature_repo.num_nodes,
        node_feature_dim=feature_repo.node_feature_dim,
        edge_feature_dim=feature_repo.edge_feature_dim,
    )

    # Training config
    train_config = config['training']
    train_batch_size = train_config['train_batch_size']
    backprop_every = train_config['backprop_every']

    # Evaluation config
    eval_batch_size = train_config.get('eval_batch_size', train_batch_size)

    num_epoch = train_config['num_epoch']
    for epoch in range(1, num_epoch + 1):
        logger.info(f'===== Start epoch {epoch} =====')
        logger.info(f'Run backprop every {backprop_every} iterations.')
        epoch_start_time = time.time()

        train_single_epoch(tgn, train_data, train_batch_size, backprop_every)
        ap, auc = evaluate_edge_prediction(tgn, val_data, eval_batch_size)
        logger.info(f'Training finished. Evaluation result: AP = {ap:.6f}, AUC = {auc:.6f}.')

        epoch_end_time = time.time()
        logger.info(f'=== Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.')
        logger.info('')
