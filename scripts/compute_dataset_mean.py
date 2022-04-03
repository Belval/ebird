import argparse
import yaml
import sys
import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ebird.model.model import Model
from ebird.model.checkpointer import Checkpointer
from ebird.datasets import build_dataset


def main(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["TRAIN"], transform=transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=16
    )

    # Taken from: https://deeplizard.com/learn/video/lu7TCu7HeYc
    num_of_pixels = len(train_dataloader) * config["TRAINING"]["BATCH_SIZE"] * 256 * 256
    total_sum_img = 0
    total_sum_features = 0
    for batch in train_dataloader:
        total_sum_img += batch[0].sum((0, 2, 3))
        total_sum_features += batch[1].sum(0)

    mean_img = total_sum_img / num_of_pixels
    mean_features = total_sum_features / (len(train_dataloader) * config["TRAINING"]["BATCH_SIZE"])

    sum_of_squared_error_img = 0
    sum_of_squared_error_features = 0
    for batch in train_dataloader: 
        sum_of_squared_error_img += ((batch[0] - mean_img[np.newaxis, :, np.newaxis, np.newaxis]).pow(2)).sum((0, 2, 3))
        sum_of_squared_error_features += ((batch[1] - mean_features[np.newaxis, :]).pow(2)).sum(0)

    std_img = torch.sqrt(sum_of_squared_error_img / num_of_pixels)
    std_features = torch.sqrt(sum_of_squared_error_features / (len(train_dataloader) * config["TRAINING"]["BATCH_SIZE"]))
    
    print(mean_img)
    print(std_img)
    print(mean_features)
    print(std_features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("eBird training script")
    parser.add_argument("-c", "--configuration", type=str, nargs="?", help="Path to your configuration file", required=True)
    args = parser.parse_args()
    with open(args.configuration, "r") as conf:
        try:
            config = yaml.safe_load(conf)
        except yaml.YAMLError as ex:
            print(ex)
            sys.exit()
    main(config)