import json
import os
import torch
from torch.utils.data import Dataset

class EBirdDataset(Dataset):
    def __init__(self, config):
        self.annotation_path = config["ANNOTATION_PATH"]
        self.image_path = config["IMAGE_PATH"]

        with open(self.annotation_path, "r") as fh:
            annotations = json.load(fh)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]