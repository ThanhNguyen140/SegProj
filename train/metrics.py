import torch


def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Dice coefficient for binary segmentation.
    y_true, y_pred: torch tensors of shape [B, H, W, D]
    Values should be 0 or 1.
    """
    assert y_true.dim() == 4 and y_pred.dim() == 4

    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)

    eps = 1e-8
    intersection = (y_true * y_pred).sum(dim=-1).float()
    union = y_true.sum(dim=-1) + y_pred.sum(dim=-1).float()

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    IoU coefficient for binary segmentation.
    y_true, y_pred: torch tensors of shape [B, H, W, D]
    Values should be 0 or 1.
    """
    assert y_true.dim() == 4 and y_pred.dim() == 4

    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)
    eps = 1e-8

    i_area = (y_true * y_pred).sum(dim=-1).float()
    u_area = torch.sum(y_true | y_pred, dim=-1).float()

    iou = (i_area + eps) / (u_area + eps)

    return iou.mean()


def precision_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps=1e-8):
    """
    Precision = TP / (TP + FP)
    y_true, y_pred: binary tensors [B, 1, D, H, W] or [B, D, H, W]
    """
    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)

    tp = (y_true * y_pred).sum(dim=-1)
    fp = torch.sum((1 - y_true) * y_pred, dim=-1)

    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean()


def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps=1e-8):
    """
    Recall = TP / (TP + FN)
    """
    y_true = y_true.flatten(1)
    y_pred = y_pred.flatten(1)

    tp = (y_true * y_pred).sum(dim=-1)
    fn = torch.sum(y_true * (1 - y_pred), dim=-1)

    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean()
