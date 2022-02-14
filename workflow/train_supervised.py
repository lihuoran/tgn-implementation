import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from data import DataBatch, Dataset, get_neighbor_finder, get_supervised_data
from model import AbsBinaryClassificationModel, AbsEmbeddingModel
from utils import (
    EarlyStopMonitor, get_module, load_model, make_logger, save_model, WorkflowContext
)
from utils.training import copy_best_model
from .evaluation import evaluate_node_classification
from .utils import prepare_workspace


def train_single_epoch(
    workflow_context: WorkflowContext,
    cls_model: AbsBinaryClassificationModel, emb_model_output: torch.Tensor,
    data: Dataset, batch_size: int,
    backprop_every: int, device: torch.device, optimizer: Optimizer, accumulated_iter_cnt: int
) -> Tuple[float, int]:
    criterion = torch.nn.BCELoss()

    iter_cnt = 0
    sample_cnt = 0
    loss = 0
    optimizer.zero_grad()
    loss_records = []

    cls_model.train()
    for batch in data.batch_generator(workflow_context, batch_size, device, desc=f'Training progress'):
        assert isinstance(batch, DataBatch)

        # Forward propagation
        src_emb = emb_model_output[sample_cnt:sample_cnt + batch.size]
        src_prob = cls_model.get_prob(src_emb)
        src_pred = src_prob.sigmoid()

        loss += criterion(src_pred, batch.labels)

        if iter_cnt % backprop_every == 0 or sample_cnt == data.n_sample:  # Back propagation
            assert isinstance(loss, torch.Tensor)
            loss /= backprop_every
            loss.backward()
            optimizer.step()
            loss_records.append(loss.item())

            workflow_context.summary_writer.add_scalar(
                'Training/Training loss', loss.item() / backprop_every, accumulated_iter_cnt + iter_cnt
            )

            loss = 0
            optimizer.zero_grad()

        sample_cnt += batch.size
        iter_cnt += 1

    return float(np.mean(loss_records)), accumulated_iter_cnt + iter_cnt


def precompute_embedding_model_output(
    workflow_context: WorkflowContext,
    emb_model: AbsEmbeddingModel,
    train_data: Dataset, train_batch_size: int,
    eval_data: Dataset, eval_batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    emb_model.epoch_start_step()
    emb_model.eval()

    #
    iter_cnt = 0
    train_embs = []
    for batch in train_data.batch_generator(
        workflow_context, train_batch_size, device, desc=f'Pre-computing progress: train'
    ):
        assert isinstance(batch, DataBatch)

        with torch.no_grad():
            src_emb, _, _ = emb_model.compute_temporal_embeddings(batch)
            train_embs.append(src_emb)

        iter_cnt += 1
    emb_model.epoch_end_step()

    #
    iter_cnt = 0
    eval_embs = []
    for batch in eval_data.batch_generator(
        workflow_context, eval_batch_size, device, desc=f'Pre-computing progress: eval'
    ):
        assert isinstance(batch, DataBatch)

        with torch.no_grad():
            src_emb, _, _ = emb_model.compute_temporal_embeddings(batch)
            eval_embs.append(src_emb)

        iter_cnt += 1
    emb_model.epoch_end_step()

    return torch.cat(train_embs), torch.cat(eval_embs)


def run_train_supervised(args: argparse.Namespace, config: dict) -> None:
    workspace_path = config['workspace_path']
    version_path = f'{workspace_path}/version__{args.version}/'
    prepare_workspace(version_path)

    workflow_context = WorkflowContext(
        logger=make_logger(os.path.join(version_path, 'log.txt')),
        dry_run=args.dry,
        summary_writer=SummaryWriter(version_path),
    )
    if workflow_context.dry_run:
        workflow_context.logger.info('Run workflow in dry mode.')

    # Read data
    data_dict, feature_repo = \
        get_supervised_data(
            workflow_context,
            workspace_path=workspace_path,
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

    # Job module & objects
    job_module = get_module(os.path.abspath(args.job_path))
    #
    get_emb_model = getattr(job_module, 'get_emb_model')
    emb_model = get_emb_model(feature_repo, device)
    assert isinstance(emb_model, AbsEmbeddingModel)
    load_model(workflow_context, emb_model, version_path=embedding_model_version_path)  # Load pretrained model
    #
    get_cls_model = getattr(job_module, 'get_cls_model')
    cls_model = get_cls_model(feature_repo.node_feature_dim(), device)
    assert isinstance(cls_model, AbsBinaryClassificationModel)

    # Pre-compute embedding model output
    emb_model.set_neighbor_finder(train_neighbor_finder)
    emb_model.eval()  # Freeze embedding model. Do not train this.
    emb_model_output_train, emb_model_output_eval = precompute_embedding_model_output(
        workflow_context,
        emb_model,
        data_dict['train'], train_batch_size,
        data_dict['eval'], eval_batch_size,
        device,
    )

    num_epoch = train_config['num_epoch']
    early_stopper = EarlyStopMonitor(max_round=train_config['early_stop'])
    optimizer = torch.optim.Adam(cls_model.parameters(), lr=lr)
    accumulated_iter_cnt = 0
    for epoch in range(1, num_epoch + 1):
        workflow_context.logger.info(f'===== Start epoch {epoch} =====')
        workflow_context.logger.info(
            f'Run backprop every {backprop_every} {"iteration" if backprop_every == 1 else "iterations"}.'
        )
        epoch_start_time = time.time()

        train_loss, accumulated_iter_cnt = train_single_epoch(
            workflow_context,
            cls_model, emb_model_output_train, data_dict['train'],
            train_batch_size, backprop_every, device, optimizer, accumulated_iter_cnt
        )
        workflow_context.logger.info(f'Training finished. Mean training loss = {train_loss:.6f}')

        save_model(workflow_context, model=cls_model, version_path=version_path, epoch=epoch)

        auc = evaluate_node_classification(
            workflow_context, cls_model, emb_model_output_eval, data_dict['eval'], eval_batch_size, device,
        )
        workflow_context.logger.info(f'Evaluation result: AUC = {auc:.6f}.')
        workflow_context.summary_writer.add_scalar('Evaluation/AUC', auc, epoch)

        if early_stopper.early_stop_check(auc):
            workflow_context.logger.info(f'No improvement over {train_config["early_stop"]} rounds, early stop.')
            workflow_context.logger.info(f'Best epoch = {early_stopper.best_epoch}')
            break

        epoch_end_time = time.time()
        workflow_context.logger.info(f'=== Epoch {epoch} finished in {epoch_end_time - epoch_start_time:.2f} seconds.')
        workflow_context.logger.info('')

    copy_best_model(version_path=version_path, best_epoch=early_stopper.best_epoch)

    load_model(workflow_context, cls_model, version_path=version_path)
    auc = evaluate_node_classification(
        workflow_context, cls_model, emb_model_output_eval, data_dict['eval'], eval_batch_size, device,
    )
    workflow_context.logger.info(f'Reload model and test. Evaluation result: AUC = {auc:.6f}.')
