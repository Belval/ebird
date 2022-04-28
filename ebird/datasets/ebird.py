import json
import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas
from PIL import Image
import torch
from torch.utils.data import Dataset

from ebird.utils.constants import STATE_CODE, EBIRD_KEYS

MEAN_FEATURES = np.array([
    2.5934e-03,  1.2967e-04,  3.1639e-02,  2.2044e-03,  1.3589e-01,
    3.8900e-02,  1.9580e-02,  2.5934e-04,  5.7054e-03,  3.2158e-02,
    1.2578e-02,  0.0000e+00,  7.9098e-03,  2.8397e-02,  1.2707e-02,
    5.4461e-03,  7.7801e-03,  5.0571e-03,  4.7977e-03,  1.6079e-02,
    3.0213e-02,  3.6566e-02,  3.5010e-02,  2.2044e-02,  1.0373e-03,
    1.3745e-02,  1.0373e-02,  4.2790e-03,  4.5384e-03,  8.9471e-03,
    2.7101e-02,  1.4263e-02,  6.4315e-02,  2.1266e-02,  1.8154e-03,
    3.9030e-02,  4.2790e-03,  3.9549e-02,  3.9808e-02,  3.5010e-03,
    8.4284e-03,  1.4263e-03,  1.0633e-02,  4.3957e-02,  1.6209e-02,
    1.0892e-02,  2.6323e-02,  5.1219e-02,  4.6680e-03,  2.6452e-02,
    5.8351e-03, -9.4119e+01,  3.9309e+01,  1.1119e+01,  1.1484e+01,
    3.4936e+01,  7.5955e+02,  2.7650e+01, -4.4385e+00,  3.2088e+01,
    1.4569e+01,  7.3311e+00,  2.0357e+01,  1.6616e+00,  8.4215e+02,
    1.0695e+02,  3.8962e+01,  3.5463e+01,  2.9475e+02,  1.3456e+02,
    2.0995e+02,  2.1075e+02,  1.9401e+03,  1.1886e+03,  1.7712e+01,
    1.6465e+01,  2.6934e+01,  5.3087e+01,  3.1694e+01,  3.8416e+01
])

STD_FEATURES = np.array([
    5.0859e-02, 1.1386e-02, 1.7503e-01, 4.6899e-02, 3.4261e-01, 1.9335e-01,
    1.3855e-01, 1.6102e-02, 7.5318e-02, 1.7641e-01, 1.1144e-01, 0.0000e+00,
    8.8583e-02, 1.6610e-01, 1.1201e-01, 7.3596e-02, 8.7860e-02, 7.0932e-02,
    6.9099e-02, 1.2578e-01, 1.7117e-01, 1.8769e-01, 1.8380e-01, 1.4682e-01,
    3.2191e-02, 1.1643e-01, 1.0132e-01, 6.5274e-02, 6.7214e-02, 9.4164e-02,
    1.6237e-01, 1.1857e-01, 2.4529e-01, 1.4426e-01, 4.2568e-02, 1.9366e-01,
    6.5274e-02, 1.9489e-01, 1.9550e-01, 5.9066e-02, 9.1418e-02, 3.7740e-02,
    1.0256e-01, 2.0499e-01, 1.2627e-01, 1.0379e-01, 1.6009e-01, 2.2043e-01,
    6.8163e-02, 1.6047e-01, 7.6164e-02, 1.8357e+01, 4.8577e+00, 5.0231e+00,
    3.5565e+00, 1.1990e+01, 2.7953e+02, 7.4698e+00, 6.5796e+00, 1.0004e+01,
    8.4097e+00, 1.0366e+01, 5.9853e+00, 6.2409e+00, 4.1372e+02, 5.4532e+01,
    2.9868e+01, 2.7111e+01, 1.5282e+02, 9.6935e+01, 1.3306e+02, 1.5101e+02,
    2.3233e+03, 4.7485e+02, 1.0637e+01, 8.8937e+00, 2.4278e+01, 2.2289e+01,
    1.5285e+01, 1.9096e+01
])

class EBirdDataset(Dataset):
    def __init__(self, config, transform=None):
        self.base_path = config["BASE_PATH"]
        self.split_file_path = os.path.join(config["BASE_PATH"], config["SPLIT_FILE"])
        self.resolution = config["RESOLUTION"]

        self.config = config

        with open(self.split_file_path, "r") as fh:
           self.samples_filename = [f.strip() for f in fh.readlines()]

        self.hotspot_info = pandas.read_csv(os.path.join(self.base_path, "hotspot_info.csv"))

        self.transform = transform

        self.samples = []
        skipped_samples = 0
        for sample in self.samples_filename:
            if sample[-3:] != ".npy":
                sample += ".npy"
            if self.hotspot_info[self.hotspot_info["hotspot_id"] == sample[:-4]].empty:
                skipped_samples += 1
                continue
            self.samples.append({
                "r": os.path.join(self.base_path, "r", sample),
                "g": os.path.join(self.base_path, "g", sample),
                "b": os.path.join(self.base_path, "b", sample),
                "nir": os.path.join(self.base_path, "nir", sample),
                "hotspot": self.hotspot_info[self.hotspot_info["hotspot_id"] == sample[:-4]],
                "label": os.path.join(self.base_path, "labels", sample),
            })
        print(f"{skipped_samples} samples were skipped")
        print(f"{len(self.samples)} samples available")

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

        startx = r.shape[2] // 2 - (self.resolution // 2)
        starty = r.shape[1] // 2 - (self.resolution // 2)

        features = []

        state_one_hot = np.zeros((len(STATE_CODE),))
        state_one_hot[STATE_CODE.index(hotspot["state_code"].iloc[0])] = 1.0
        features.append(state_one_hot)

        lon_lat = np.zeros((2,))
        lon_lat[0] = hotspot["lon"].iloc[0]
        lon_lat[1] = hotspot["lat"].iloc[0]
        features.append(lon_lat)

        other_features = np.zeros((len(EBIRD_KEYS),))
        for i, k in enumerate(EBIRD_KEYS):
            other_features[i] = hotspot[k].iloc[0]
        features.append(other_features)

        feature_vector = np.nan_to_num((np.concatenate(features) - MEAN_FEATURES) / STD_FEATURES)

        img = np.moveaxis(np.concatenate([
            r[:, starty:starty+self.resolution, startx:startx+self.resolution],
            g[:, starty:starty+self.resolution, startx:startx+self.resolution],
            b[:, starty:starty+self.resolution, startx:startx+self.resolution],
            nir[:, starty:starty+self.resolution, startx:startx+self.resolution],
        ]).astype('float32'), 0, -1)

        if self.transform is not None:
            img = self.transform(img)

        return img, feature_vector, label, sample["label"]
