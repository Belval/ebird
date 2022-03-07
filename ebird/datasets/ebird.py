import json
import os
import numpy as np
import pandas
from PIL import Image
import torch
from torch.utils.data import Dataset

class EBirdDataset(Dataset):
    def __init__(self, config, transform=None):
        self.base_path = config["BASE_PATH"]
        self.split_file_path = os.path.join(config["BASE_PATH"], config["SPLIT_FILE"])

        with open(self.split_file_path, "r") as fh:
           self.samples_filename = [f.strip() for f in fh.readlines()]

        self.hotspot_info = pandas.read_csv(os.path.join(self.base_path, "hotspot_info.csv"))

        self.samples = []
        for sample in self.samples_filename:
            self.samples.append({
                "rgb": os.path.join(self.base_path, "rgb", sample[:-3] + "jpg"),
                "nir": os.path.join(self.base_path, "nir", sample),
                "hotspot": self.hotspot_info[self.hotspot_info["hotspot_id"] == sample],
                "label": os.path.join(self.base_path, "labels", sample),
            })

        print("blah")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = Image.open(sample["rgb"])
        nir = np.load(sample["nir"])
        hotspot = sample["hotspot"]
        label = np.load(sample["label"])

        return self.samples[idx]