import torch
import torch.nn as nn
import torch.nn.functional as F

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss for Monocular Depth Estimation.
    Computes scale-invariant depth error in log space.
    """
    def __init__(self, variance_focus: float = 0.85):
        super(SILogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0.001) & (gt < 80.0) & ~torch.isnan(gt) & ~torch.isinf(gt)
            
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.clamp(pred[mask], min=1e-3, max=80.0)
        gt_valid = torch.clamp(gt[mask], min=1e-3, max=80.0)

        log_diff = torch.log(pred_valid) - torch.log(gt_valid)
        
        # SILog Loss Formula: E(log_diff^2) - variance_focus * (E(log_diff))^2
        loss = torch.mean(log_diff ** 2) - self.variance_focus * (torch.mean(log_diff) ** 2)
        return torch.sqrt(loss + 1e-8)

class DepthGradientLoss(nn.Module):
    """
    Depth Gradient Loss for preserving sharp object boundaries and structural edges.
    """
    def __init__(self):
        super(DepthGradientLoss, self).__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = (gt > 0.001) & (gt < 80.0)
            
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Horizontal & Vertical spatial gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        loss_dx = torch.mean(torch.abs(pred_dx - gt_dx)[mask_dx]) if mask_dx.any() else 0.0
        loss_dy = torch.mean(torch.abs(pred_dy - gt_dy)[mask_dy]) if mask_dy.any() else 0.0

        return loss_dx + loss_dy

class CombinedDepthLoss(nn.Module):
    """
    Combined Loss: SILog Loss + 0.5 * Gradient Loss
    """
    def __init__(self, grad_weight: float = 0.5):
        super(CombinedDepthLoss, self).__init__()
        self.silog_loss = SILogLoss()
        self.grad_loss = DepthGradientLoss()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        silog = self.silog_loss(pred, gt, mask)
        grad = self.grad_loss(pred, gt, mask)
        return silog + self.grad_weight * grad
