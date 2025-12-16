"""
TransOmni - Loss Functions for Multi-Task Segmentation
Adapted from TransDoDNet's loss functions for 2D histopathology images

Includes:
- DiceLoss4MOTS: Dice loss for multi-organ/tissue segmentation
- CELoss4MOTS: Cross-entropy loss with ignore index support
- Combined loss for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""
    
    def __init__(self, smooth=1.0, apply_nonlin=None, batch_dice=False, 
                 do_bg=True, smooth_in_nom=True, background_weight=1, rebalance_weights=None):
        super(SoftDiceLoss, self).__init__()
        
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.smooth_in_nom = smooth_in_nom
        self.background_weight = background_weight
        self.rebalance_weights = rebalance_weights

    def forward(self, x, y, loss_mask=None):
        """Compute Soft Dice Loss.
        
        Args:
            x: Predictions [B, C, H, W]
            y: Ground truth [B, C, H, W] or [B, H, W]
            loss_mask: Optional mask for weighted loss
            
        Returns:
            Dice loss value
        """
        shp_x = x.shape
        
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        
        # Ensure y has same shape as x
        if len(y.shape) == len(x.shape) - 1:
            y = y.unsqueeze(1)
        
        tp, fp, fn = self.get_tp_fp_fn(x, y, axes, loss_mask, False)
        
        if self.smooth_in_nom:
            dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        else:
            dc = (2 * tp) / (2 * tp + fp + fn + self.smooth)
        
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        
        dc = dc.mean()
        
        return 1 - dc

    def get_tp_fp_fn(self, net_output, gt, axes, mask=None, square=False):
        """Compute true positives, false positives, false negatives."""
        if mask is not None:
            net_output = net_output * mask
            gt = gt * mask
        
        if square:
            tp = (net_output ** 2 * gt ** 2).sum(dim=axes)
            fp = (net_output ** 2 * (1 - gt) ** 2).sum(dim=axes)
            fn = ((1 - net_output) ** 2 * gt ** 2).sum(dim=axes)
        else:
            tp = (net_output * gt).sum(dim=axes)
            fp = (net_output * (1 - gt)).sum(dim=axes)
            fn = ((1 - net_output) * gt).sum(dim=axes)
        
        return tp, fp, fn


class DiceLoss4MOTS(nn.Module):
    """Dice Loss for Multi-Organ/Tissue Segmentation.
    
    Handles multi-task scenarios where some classes may be ignored
    (marked with -1 in the label).
    """
    
    def __init__(self, num_classes=2):
        super(DiceLoss4MOTS, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, targets):
        """Compute Dice Loss.
        
        Args:
            preds: Predictions [B, num_classes, H, W]
            targets: Ground truth [B, num_classes, H, W] or [B, H, W]
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to predictions
        preds = torch.sigmoid(preds)
        
        # Handle 2D targets (no channel dimension)
        if len(targets.shape) == 3:
            # Expand targets to match predictions
            targets = targets.unsqueeze(1)
        
        loss = 0.0
        count = 0
        
        for c in range(self.num_classes):
            if c < targets.shape[1]:
                target_c = targets[:, c]
            else:
                target_c = targets[:, 0]
            
            pred_c = preds[:, c]
            
            # Skip ignored labels (marked as -1)
            valid_mask = (target_c >= 0).float()
            
            if valid_mask.sum() > 0:
                # Compute dice for this class
                target_c = target_c.clamp(0, 1)  # Ensure binary
                pred_c = pred_c * valid_mask
                target_c = target_c * valid_mask
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                
                dice = (2.0 * intersection + 1.0) / (union + 1.0)
                loss += (1.0 - dice)
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CELoss4MOTS(nn.Module):
    """Cross-Entropy Loss for Multi-Organ/Tissue Segmentation.
    
    Handles binary segmentation with ignore index support.
    """
    
    def __init__(self, num_classes=2, ignore_index=255):
        super(CELoss4MOTS, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, preds, targets, weights=None):
        """Compute Binary Cross-Entropy Loss.
        
        Args:
            preds: Predictions [B, num_classes, H, W]
            targets: Ground truth [B, num_classes, H, W] or [B, H, W]
            weights: Optional edge weights [B, H, W]
            
        Returns:
            BCE loss value
        """
        # Handle 2D targets
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        
        loss = 0.0
        count = 0
        
        for c in range(self.num_classes):
            if c < targets.shape[1]:
                target_c = targets[:, c]
            else:
                target_c = targets[:, 0]
            
            pred_c = preds[:, c]
            
            # Create valid mask (exclude ignored and invalid labels)
            valid_mask = (target_c >= 0) & (target_c != self.ignore_index)
            
            if valid_mask.sum() > 0:
                target_c = target_c.clamp(0, 1).float()
                
                # Apply weights if provided
                if weights is not None:
                    bce = F.binary_cross_entropy_with_logits(
                        pred_c, target_c, reduction='none'
                    )
                    bce = bce * valid_mask.float()
                    # Add extra weight to edges
                    bce = bce * (1.0 + weights)
                    loss += bce.sum() / (valid_mask.float().sum() + 1e-6)
                else:
                    bce = F.binary_cross_entropy_with_logits(
                        pred_c[valid_mask], target_c[valid_mask]
                    )
                    loss += bce
                
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CombinedLoss(nn.Module):
    """Combined Dice + Cross-Entropy Loss.
    
    Commonly used combination for segmentation tasks.
    """
    
    def __init__(self, num_classes=2, dice_weight=1.0, ce_weight=1.0, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss4MOTS(num_classes)
        self.ce_loss = CELoss4MOTS(num_classes, ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, preds, targets, weights=None):
        """Compute combined loss.
        
        Args:
            preds: Predictions [B, num_classes, H, W]
            targets: Ground truth
            weights: Optional edge weights
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets, weights)
        return self.dice_weight * dice + self.ce_weight * ce
