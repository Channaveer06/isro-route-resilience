import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Calculate AMP-safe BCE with logits directly from inputs
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Apply sigmoid for the Dice calculation
        inputs_sig = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def calculate_iou(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou
