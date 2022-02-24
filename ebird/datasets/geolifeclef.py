import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class GeoLifeCLEFDataset(Dataset):
    def __init__(self, config, transform=None):
        self.annotation_path = config["ANNOTATION_PATH"]
        self.image_path = config["IMAGE_PATH"]
        self.transform = transform

        with open(self.annotation_path, "r") as fh:
            annotations = json.load(fh)
            self.annotations = {annot['id']:annot for annot in annotations["annotations"]}
            self.images = {img['id']:img for img in annotations["images"]}

        self.samples = []
        for _, annot in self.annotations.items():
            if self.images[annot["image_id"]]["country"] != "us":
                continue
            self.samples.append((
                os.path.join(self.image_path, self.images[annot["image_id"]]["file_name"]),
                os.path.join(self.image_path, self.images[annot["image_id"]]["file_name_alti"]),
                annot["category_id"],
            ))
        print(f"Dataset {self.annotation_path} contains {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fn, img_alti_fn, gt = self.samples[idx]
        img = np.load(img_fn)
        img_alti = np.load(img_alti_fn)

        img = np.concatenate([img, img_alti[:, :, np.newaxis]], axis=-1)

        #img = np.moveaxis(img, -1, 0)

        img = img.astype('float32')

        if self.transform is not None:
            img = self.transform(img)

        return img, gt
