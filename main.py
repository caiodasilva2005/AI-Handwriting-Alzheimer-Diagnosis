

from data import DarwinDownloader, SampleType
from model import HandwritingAlzheimerDataset, HandwritingAlzheimerDatasetDisplayer

def main():
    downloader = DarwinDownloader()
    annotations, image_dir =  downloader.generateAnnotations(sampleType=SampleType.RGB_ON_PAPER, taskNumber=2)

    dataset = HandwritingAlzheimerDataset(annotations_file=annotations, img_dir=image_dir)
    model = HandwritingAlzheimerDatasetDisplayer(dataset)
    
    model.displayBatch()

    return 0

if __name__ == "__main__":
    main()