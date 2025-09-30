import torch


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Dice loss for binary segmentation.
    y_true, y_pred: torch tensors of shape [B, H, W, D]
    Values should be 0 or 1 for y_true and probability [0,1] for y_pred .
    """
    assert y_true.dim() == 4 and y_pred.dim() == 4

    y_true = y_true.sigmoid().flatten(1)
    y_pred = y_pred.flatten(1)

    eps = 1e-8
    intersection = (y_true * y_pred).sum(dim=-1).float()
    union = y_true.sum(dim=-1) + y_pred.sum(dim=-1).float()

    dice_loss = 1 - (2.0 * intersection + eps) / (union + eps)

    return dice_loss.mean()
