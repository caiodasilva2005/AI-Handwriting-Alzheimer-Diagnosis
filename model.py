from data import HandwritingAlzheimerDataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
class ResnetCNN(nn.Module):
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
    
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 71 * 71, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HandwritingModel:
    def __init__(self, net, name):
        self.net = net
        self.name = name
        self.dataloader = None

    def _updatedWithSaved(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Given file to update model does not exist.")
        self.net.load_state_dict(torch.load(model_path, weights_only=True))

    def createDataloader(self, annotations, img_dir, transform, batch_size=32, shuffle=True):
        dataset = HandwritingAlzheimerDataset(annotations_file=annotations, img_dir=img_dir, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train(self, trainer, num_iterations=5, output_dir="data"):
        if self.dataloader is None:
            raise Exception("Dataloader not created. Call createDataloader before training.")
        trainer.train(self.net, self.dataloader, num_iterations)

        output_path = os.path.join(output_dir, f"{self.name}.pth")
        trainer.saveModel(self.net, output_path)

        return output_path
    
    def getParameters(self):
        return self.net.parameters()

    def test(self, tester, model_path_dir):
        self._updatedWithSaved(f"{model_path_dir}/{self.name}.pth")

        if self.dataloader is None:
            raise Exception("Dataloader not created. Call createDataloader before testing.")
        tester.test(self.net, self.dataloader)




    