

from data import DarwinDownloader, SampleType
from data import HandwritingAlzheimerDataset
from model import CNN
from test import Tester
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch


transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    # Normalized to values corresponding to the ImageNet dataset
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225))])

def main():
    downloader = DarwinDownloader()
    annotations, image_dir =  downloader.generateAnnotations(sampleType=SampleType.RGB_ON_PAPER, taskNumber=2)

    dataset = HandwritingAlzheimerDataset(annotations_file=annotations, img_dir=image_dir, transform=transform)
    testloader = DataLoader(dataset, batch_size=32, shuffle=False)

    net = CNN()

    tester = Tester(net, testloader)
    tester.updatedWithSaved("data/model.pth")

    tester.test()
    
    return 0

if __name__ == "__main__":
    main()

    