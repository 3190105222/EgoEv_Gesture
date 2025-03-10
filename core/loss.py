import torch.nn as nn

class CustomClassificationLoss(nn.Module):
    def __init__(self):
        super(CustomClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_classes, gt_classes):
        return self.criterion(pred_classes, gt_classes)

