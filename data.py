
import os
from os import path
from enum import Enum

import pandas as pd
import kagglehub
import shutil

class SampleType(Enum):
    RGB_ON_PAPER = "rgb_on_paper"
    OFFLINE = "offline"

class PatientType(Enum):
    HEALTHY = ("HC", 0)
    ALZHEIMER = ("AD", 1)

# Download latest version
class DarwinDownloader:
    def __init__(self):
        self.dataset_path = self.downloadDarwinDataset()
        self.tasks = {
            SampleType.RGB_ON_PAPER: (2, 3, 4, 5, 21, 24),
            SampleType.OFFLINE: (2, 3, 4, 5, 21, 24)
        }

    def _formatTaskNumber(self, taskNumber):
        return f"TASK_{taskNumber:02d}"

    # returns the path the DARWIN dataset is downloaded to
    def downloadDarwinDataset(self):
        try: 
            path = kagglehub.dataset_download("tizianadalessandro/darwin-i")
        except Exception as e:
            raise Exception(f"Error downloading dataset: {e}")
    
        return path
    
    def generateAnnotations(self, sampleType, taskNumber, annoationsDir="data/annotations"):
        if taskNumber not in self.tasks[sampleType]:
            raise ValueError(f"Invalid task number {taskNumber} for sample type {sampleType}")

        task_path = path.join(self.dataset_path, sampleType.value, sampleType.value, self._formatTaskNumber(taskNumber))
        
        os.makedirs(annoationsDir, exist_ok=True)
        
        image_dir_path = path.join(annoationsDir, self._formatTaskNumber(taskNumber))
        os.makedirs(image_dir_path, exist_ok=True)

        rows = []
        for label_name, label in ((PatientType.HEALTHY.value[0], PatientType.HEALTHY.value[1]), 
                                  (PatientType.ALZHEIMER.value[0], PatientType.ALZHEIMER.value[1])):
            class_dir = path.join(task_path, label_name)
            for filename in sorted(os.listdir(class_dir)):
                rows.append((filename, label))
                shutil.copy(path.join(class_dir, filename), image_dir_path)

        annotations_path = path.join(annoationsDir, "annotations.csv")
        print(f"Saving annotations to {annotations_path}")
        pd.DataFrame(rows, columns=["filepath", "label"]).to_csv(annotations_path, index=False)
        return (annotations_path, image_dir_path)
