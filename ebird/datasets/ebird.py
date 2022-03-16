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

        self.transform = transform

        self.samples = []
        for sample in self.samples_filename:
            self.samples.append({
                "r": os.path.join(self.base_path, "r", sample),
                "g": os.path.join(self.base_path, "g", sample),
                "b": os.path.join(self.base_path, "b", sample),
                "nir": os.path.join(self.base_path, "nir", sample),
                "hotspot": self.hotspot_info[self.hotspot_info["hotspot_id"] == sample],
                "label": os.path.join(self.base_path, "labels", sample),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        r = np.load(sample["r"])
        g = np.load(sample["g"])
        b = np.load(sample["b"])
        nir = np.load(sample["nir"])
        hotspot = sample["hotspot"]
        label = (np.load(sample["label"]) > 0.0).astype(np.float32)

        startx = r.shape[2] // 2 - (256 // 2)
        starty = r.shape[1] // 2 - (256 // 2)

        img = np.moveaxis(np.concatenate([
            r[:, starty:starty+256, startx:startx+256],
            g[:, starty:starty+256, startx:startx+256],
            b[:, starty:starty+256, startx:startx+256],
            nir[:, starty:starty+256, startx:startx+256],
        ]).astype('float32'), 0, -1)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
