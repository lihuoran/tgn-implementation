import torch

from model import AbsBinaryClassificationModel
from model.simple_cls_model import SimpleClassificationModel


def get_cls_model(input_dim: int, device: torch.device) -> AbsBinaryClassificationModel:
    return SimpleClassificationModel(input_dim, device)
