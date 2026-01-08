import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(y_pred: torch.Tensor,y_true: torch.Tensor):
    """
    Dice loss for binary segmentation.
    y_true, y_pred: torch tensors of shape [B, H, W, D]
    Values should be 0 or 1 for y_true and logits for y_pred .
    """
    assert y_true.dim() == 4 and y_pred.dim() == 4

    y_true = y_true.flatten(1)
    prob = y_pred.sigmoid().flatten(1)

    eps = 1e-8
    intersection = (y_true * prob).sum(dim=-1).float()
    union = y_true.sum(dim=-1).float() + prob.sum(dim=-1).float()

    dice_loss = 1 - ((2.0 * intersection + eps) / (union + eps))
    #print('dice loss:',dice_loss.mean().item())
    return dice_loss.mean()

def sigmoid_focal_loss(
    y_pred,
    y_true,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Args:
        y_pred: A float tensor of arbitrary shape.
                The predictions for each example.
        y_true: A float tensor with the same shape as y_pred. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        focal loss tensor
    """
    prob = y_pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    p_t = prob * y_true + (1 - prob) * (1 - y_true)
    p_t = torch.clamp(p_t, min=1e-6, max=1.0 - 1e-6)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss = alpha_t * loss
    #print('focal loss:',focal_loss.mean().item())
    return focal_loss.mean()
    
def binary_cross_entropy(y_pred,y_label):
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(y_pred, y_label)
    #print('bce loss:',bce_loss.item())
    return bce_loss

class MaskLoss(nn.Module):
    """
    Mask loss function for SAM2.
    """

    def __init__(
        self,
        alpha_focal_loss: float = 0.75,
        gamma_focal_loss: float = 2.0,
        scale_focal_loss: float = 0.3,
        scale_dice_loss: float = 0.7,
        scale_bce_loss:float = 0.0,
    ):
        super().__init__()

        self.scale_focal_loss = scale_focal_loss
        self.scale_dice_loss = scale_dice_loss
        self.scale_bce_loss = scale_bce_loss

        self.alpha_focal_loss = alpha_focal_loss
        self.gamma_focal_loss = gamma_focal_loss

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            pred_ious: A float tensor containing the predicted IoUs scores per mask
            num_objects: Number of objects in the batch
            loss_on_multimask: True if multimask prediction is enabled
        Returns:
            A Tensor of loss for each example in the batch.
        """
        loss = 0.0
        if inputs.shape != targets.shape:
            raise ValueError(
                f"Inputs and targets must have the same shape. Got {inputs.shape} and {targets.shape}."
            )
        loss += self.scale_focal_loss * sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha_focal_loss,
            gamma=self.gamma_focal_loss,
        )
        loss += self.scale_bce_loss * binary_cross_entropy(inputs, targets)
        loss += self.scale_dice_loss * dice_loss(inputs, targets)
    
        return loss.to(dtype = torch.float32)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma = 1.25, smooth=1e-6):
        """
        alpha > beta  → more penalty on FN → focuses on recall (positives).
        beta > alpha  → more penalty on FP → focuses on precision.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
    def forward(self, y_pred, y_true):
        # y_pred: (batch, 1, H, W, D) or (batch, 1, H, W)
        # y_true: same shape, binary {0,1}
        y_pred = torch.sigmoid(y_pred)   # ensure probs
        y_pred = y_pred.flatten(1)
        y_true = y_true.flatten(1)

        TP = (y_pred * y_true).sum(axis = -1)
        FP = ((1 - y_true) * y_pred).sum(axis = -1)
        FN = (y_true * (1 - y_pred)).sum(axis = -1)

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        loss = torch.pow((1-tversky_index), self.gamma)
        return loss.mean()
