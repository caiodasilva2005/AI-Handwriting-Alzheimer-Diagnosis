
import torch
import os

class Tester:
    def __init__(self, model, testloader):
        self.model = model
        self.testloader = testloader

    def updatedWithSaved(self, path):
        if not os.path.exists(path):
            raise Exception("Given file to update model does not exist.")
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def test(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

