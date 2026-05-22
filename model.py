import torch.nn as nn
from torchvision import models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Linear probe: ResNet18 is pretrained on ImageNet. We freeze the entire backbone and
        # train only a new fc head for healthy vs. Alzheimer's classification. Freezing first
        # leaves the new nn.Linear (created after) trainable by default.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def train(self, mode: bool = True):
        # Force the frozen backbone to stay in eval mode so BatchNorm running stats
        # don't drift on the small target set; only the new head follows `mode`.
        super().train(mode)
        self.backbone.eval()
        self.backbone.fc.train(mode)
        return self

    def forward(self, x):
        return self.backbone(x)
