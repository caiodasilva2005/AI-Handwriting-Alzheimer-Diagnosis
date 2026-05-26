from data import DarwinDownloader, SampleType
from data import HandwritingAlzheimerDataset
from model import ResnetCNN, CustomCNN, HandwritingModel
from test import Tester
from train import Trainer
from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.optim as optim


NUM_ITERATIONS = 100
DATA_DIR_PATH = "data"

# ResNet18 backbone expects 224x224 inputs normalized with ImageNet statistics,
# since we're using its pretrained weights.
resnet_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225))])

# CustomCNN trains from scratch on the original 299x299 image size,
# so no ImageNet normalization.
custom_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True)])

def train_resnet(model):

    trainable = [p for p in model.getParameters() if p.requires_grad]
    optimizer = optim.SGD(trainable, lr=1e-2, momentum=0.9)

    trainer = Trainer(nn.CrossEntropyLoss(), optimizer)
    model.train(trainer, NUM_ITERATIONS, DATA_DIR_PATH)

    print("Finished training ResNet CNN.")

def train_custom(model):

    optimizer = optim.SGD(model.getParameters(), lr=1e-4, momentum=0.9)
    
    trainer = Trainer(nn.CrossEntropyLoss(), optimizer)
    model.train(trainer, NUM_ITERATIONS, DATA_DIR_PATH)

    print("Finished training custom CNN.")

def run_test(model):
    tester = Tester()
    model.test(tester, DATA_DIR_PATH)

def main():

    # annotations, image_dir = DarwinDownloader().generateAnnotations(SampleType.RGB_ON_PAPER, 2)

    resnet_model = HandwritingModel(ResnetCNN(), "resnet")
    custom_model = HandwritingModel(CustomCNN(), "custom")

    resnet_model.createDataloader("data/annotations/annotations.csv", "data/annotations/TASK_02", resnet_transform)
    custom_model.createDataloader("data/annotations/annotations.csv", "data/annotations/TASK_02", custom_transform)
    
    actions = {
        "1": ("Train ResNet CNN", train_resnet, resnet_model),
        "2": ("Train Custom CNN", train_custom, custom_model),
        "3": ("Test ResNet CNN", run_test, resnet_model),
        "4": ("Test Custom CNN", run_test, custom_model),
    }
    while True:
        print("\n--- Handwriting Alzheimer Classifier ---")
        for key, (label, _, _) in actions.items():
            print(f"  {key}) {label}")
        print("  q) Quit")
        choice = input("Select: ").strip().lower()
        if choice == "q":
            return 0
        action = actions.get(choice)
        if action is None:
            print("Invalid choice.")
            continue
        action[1](action[2])


if __name__ == "__main__":
    main()
