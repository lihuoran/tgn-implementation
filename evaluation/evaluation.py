from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from data import Dataset
from model import AbsBinaryClassificationModel, AbsEmbeddingModel
from utils import RandomNodeSelector


def evaluate_edge_prediction(
    emb_model: AbsEmbeddingModel, data: Dataset, batch_size: int, seed: int, device: torch.device,
    dry_run: bool = False
) -> Tuple[float, float]:
    random_node_selector = RandomNodeSelector(data.dst_ids, seed=seed)
    val_ap = []
    val_auc = []

    emb_model.eval()
    with torch.no_grad():
        batch_num = data.get_batch_num(batch_size)
        batch_generator = data.batch_generator(batch_size, device)

        iter_cnt = 0
        for pos_batch in tqdm(batch_generator, total=batch_num, desc=f'Evaluation progress', unit='batch'):
            batch_size = pos_batch.size
            pos_batch.neg_ids = torch.from_numpy(random_node_selector.sample(batch_size)).long()
            pos_prob, neg_prob = emb_model.compute_edge_probabilities(pos_batch)

            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

            iter_cnt += 1
            if dry_run and iter_cnt == 5:
                break
    return float(np.mean(val_ap)), float(np.mean(val_auc))


def evaluate_node_classification(
    emb_model: AbsEmbeddingModel, cls_model: AbsBinaryClassificationModel,
    data: Dataset, batch_size: int, seed: int, device: torch.device,
    dry_run: bool = False
) -> float:
    raise NotImplementedError
