
from model import CNN
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from data import DarwinDownloader, HandwritingAlzheimerDataset, SampleType

NUM_ITERATIONS = 2
class Trainer:
    def __init__(self, model, loss_function, optimizer, trainloader):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.trainloader = trainloader

    def train(self, num_iterations):
        for _ in range(num_iterations):  
            for data in self.trainloader:
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()

                self.optimizer.step()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    # Normalized to values corresponding to the ImageNet dataset
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225))])

net = CNN()
lossFunction = nn.CrossEntropyLoss()

# Momentum: 0.9 (allows for the optimizer to accelerate in the correct direction of the current step's gradient)
trainable = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.SGD(trainable, lr=1e-2, momentum=0.9)

downloader = DarwinDownloader()
annotations, image_dir = downloader.generateAnnotations(SampleType.RGB_ON_PAPER, 2)
trainset = HandwritingAlzheimerDataset(annotations_file=annotations, img_dir=image_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

trainer = Trainer(net, lossFunction, optimizer, trainloader)
trainer.train(NUM_ITERATIONS)
trainer.save_model("data/model.pth")

print('Finished Training')
