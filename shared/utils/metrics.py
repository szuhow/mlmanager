import torch
import torch.nn as nn

class Dice(nn.Module):
    def __init__(self, smooth=1.0):
        super(Dice, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class IoU(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoU, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + self.smooth) / (union + self.smooth)
