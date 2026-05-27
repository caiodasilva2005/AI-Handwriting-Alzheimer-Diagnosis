"""
Loads image folders per task 
"""

import os
from os import path
from enum import Enum
import pandas as pd
import kagglehub
import shutil
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode

DATASET_IMAGES_DIR = "dataset_offline - task_2_25"
PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), "..", ".."))                                                                                                                                           
DEFAULT_ANNOTATIONS_DIR = path.join(PROJECT_ROOT, "data", "annotations") 

class SampleType(Enum):
    TEST = "Test"
    TRAIN = "Train"
    VAL = "Val"
class PatientType(Enum):
    HEALTHY = ("HC", 0)
    ALZHEIMER = ("PT", 1)

# Download latest version
class DarwinDownloader:
    def __init__(self):
        self.dataset_path = self._downloadDarwinDataset()
        
    def _formatTaskNumber(self, taskNumber):
        return f"TASK_{taskNumber:02d}"

    # returns the path the DARWIN dataset is downloaded to
    def _downloadDarwinDataset(self):
        try: 
            path = kagglehub.dataset_download("tizianadalessandro/darwin-i")
        except Exception as e:
            raise Exception(f"Error downloading dataset: {e}")
    
        return path
    
    def generateAnnotations(self, sampleType, foldCount=5, annoationsDir=DEFAULT_ANNOTATIONS_DIR):

        if foldCount < 1 or foldCount > 5:
            raise Exception("Invalid Fold Count. Must be from 1 - 5.")

        os.makedirs(annoationsDir, exist_ok=True)
        image_dir_path = path.join(annoationsDir, sampleType.value)
        os.makedirs(image_dir_path, exist_ok=True)

        rows = []
        # Tasks Range from TASK_02 to TASK_25
        for taskNumber in range(2, 26):
            # Each TASK has up to folds from Fold1 - Fold5
            for foldNum in range(1, foldCount + 1):
                task_path = path.join(self.dataset_path, DATASET_IMAGES_DIR, DATASET_IMAGES_DIR, self._formatTaskNumber(taskNumber), f"Fold{foldNum}", sampleType.value)

                for label_name, label in ((PatientType.HEALTHY.value[0], PatientType.HEALTHY.value[1]),
                                        (PatientType.ALZHEIMER.value[0], PatientType.ALZHEIMER.value[1])):
                    class_dir = path.join(task_path, label_name)
                    for filename in sorted(os.listdir(class_dir)):
                        stem, ext = path.splitext(filename)
                        new_filename = f"{stem}_Fold{foldNum}{ext}"
                        rows.append((new_filename, label))
                        shutil.copy(path.join(class_dir, filename), path.join(image_dir_path, new_filename))

        annotations_path = path.join(annoationsDir, "annotations.csv")
        print(f"Saving annotations to {annotations_path}")
        pd.DataFrame(rows, columns=["filepath", "label"]).to_csv(annotations_path, index=False)
        return (annotations_path, image_dir_path)

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
        image = decode_image(img_path, mode=ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label