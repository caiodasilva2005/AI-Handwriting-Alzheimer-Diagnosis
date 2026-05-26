import os
import torch


class Trainer:
    def __init__(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, model, dataloader, num_iterations):
        for _ in range(num_iterations):
            for data in dataloader:
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()

                self.optimizer.step()

    def saveModel(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)




