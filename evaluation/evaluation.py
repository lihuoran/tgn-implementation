import dataclasses
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data.data import Dataset
from model.abs_model import AbsModel
from utils.training import RandomNodeSelector


def evaluate_edge_prediction(model: AbsModel, data: Dataset, batch_size: int, seed: int) -> Tuple[float, float]:
    random_node_selector = RandomNodeSelector(data.dst_ids, seed=seed)
    val_ap = []
    val_auc = []
    with torch.no_grad():
        for pos_batch in data.batch_generator(batch_size):
            neg_batch = dataclasses.replace(pos_batch)
            neg_batch.dst_ids = torch.from_numpy(random_node_selector.sample(neg_batch.size))
            pos_prob = model.compute_edge_probabilities(pos_batch)
            neg_prob = model.compute_edge_probabilities(neg_batch)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(pos_batch.size), np.zeros(neg_batch.size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return float(np.mean(val_ap)), float(np.mean(val_auc))
