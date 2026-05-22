

from data import DarwinDownloader, SampleType
from data import HandwritingAlzheimerDataset
from model import CNN
from test import Tester

def main():
    downloader = DarwinDownloader()
    annotations, image_dir =  downloader.generateAnnotations(sampleType=SampleType.RGB_ON_PAPER, taskNumber=2)

    dataset = HandwritingAlzheimerDataset(annotations_file=annotations, img_dir=image_dir)

    net = CNN()
    
    tester = Tester(net, dataset)
    tester.updatedWithSaved("data/model.pth")

    tester.test()
    
    return 0

if __name__ == "__main__":
    main()

    