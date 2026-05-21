
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import decode_image
import matplotlib.pyplot as plt


# Class for model that will be trained to diagnose Alzheimer's disease from handwriting samples
class HandwritingAlzheimerDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# TODO: delete this is temporary for debugging
class HandwritingAlzheimerDatasetDisplayer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self._iterator = iter(self.dataloader)
        self._batch = None
        self._batch_idx = 0

    def _nextSample(self):
        if self._batch is None or self._batch_idx >= self._batch[0].size(0):
            try:
                self._batch = next(self._iterator)
            except StopIteration:
                self._iterator = iter(self.dataloader)
                self._batch = next(self._iterator)
            self._batch_idx = 0
        features, labels = self._batch
        img, label = features[self._batch_idx], labels[self._batch_idx]
        self._batch_idx += 1
        return img, label

    def displayBatch(self):
        for _ in range(len(self.dataset)):
            img, label = self._nextSample()
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f"Label: {label}")
            plt.show()

