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
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

from ebird.model.model import Model
from ebird.model.checkpointer import Checkpointer
from ebird.datasets import build_dataset
from ebird.datasets.ebird import MEAN_FEATURES, STD_FEATURES
from ebird.utils.constants import STATE_CODE, EBIRD_KEYS

def main(config):
    output_path = os.path.join(config["OUTPUT_DIR"], str(time.time()))
    print(output_path)
    os.makedirs(output_path)
    device = "cuda"

    model = Model(config["MODEL"]).to(device)

    if config["DATASET"]["TRAIN"]["TYPE"] == "GeoLifeCLEFDataset":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[111.9393, 121.3099, 113.0863, 140.8277],
                std=[51.5302,  45.5618,  41.4096,  54.2996],
            ),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[887.4186,  929.5382,  688.8167, 2818.0081],
                std=[873.2404,  734.9333,  734.1111, 1351.0328],
            ),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

    if "RESUME" in config and config["RESUME"]:
        checkpoint = torch.load(config["RESUME"])
        new_state_dict = {
            k:(v if v.size() == model.state_dict()[k].size() else model.state_dict()[k])
            for k, v in zip(model.state_dict().keys(), checkpoint["model_state_dict"].values())
        }
        model.load_state_dict(new_state_dict, strict=False)

    val_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["VALIDATION"], transform=transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=16
    )

    geometry_gt = {}
    geometry_predicted = {}
    for i, (input_images, input_features, targets, filename) in enumerate(val_dataloader):
        outputs, _ = model(input_images.to(device), input_features.to(device))
        for j, v in torch.nonzero(targets).numpy().tolist():
            geometry_gt[v] = geometry_gt.get(v, []) + [Point(
                    input_features[j][len(STATE_CODE)] * STD_FEATURES[len(STATE_CODE)] + MEAN_FEATURES[len(STATE_CODE)],
                    input_features[j][len(STATE_CODE) + 1] * STD_FEATURES[len(STATE_CODE) + 1] + MEAN_FEATURES[len(STATE_CODE) + 1],
                )]
        
        for j, v in torch.nonzero(outputs > 0.5).detach().cpu().numpy().tolist():
            if v == 36:
                if targets[j, v]:
                    print(f"Good: {filename[j]}")
                else:
                    print(f"Bad: {filename[j]}")
            geometry_predicted[v] = geometry_predicted.get(v, []) + [Point(
                input_features[j][len(STATE_CODE)] * STD_FEATURES[len(STATE_CODE)] + MEAN_FEATURES[len(STATE_CODE)],
                input_features[j][len(STATE_CODE) + 1] * STD_FEATURES[len(STATE_CODE) + 1] + MEAN_FEATURES[len(STATE_CODE) + 1],
            )]

    for i in range(config["MODEL"]["CLASSIFICATION_HEAD"]["OUTPUT_FEATURES"]):
        if i not in geometry_gt or i not in geometry_predicted:
            continue

        df = pd.DataFrame()
        gdf = GeoDataFrame(df, geometry=geometry_gt[i])
        world = gpd.read_file('geopandas_data/usa-states-census-2014.shp')
        fig = gdf.plot(
            ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15
        ).get_figure()
        fig.axes[0].axis("off")
        fig.savefig(os.path.join(output_path, f"{i}_gt.png"), bbox_inches='tight', transparent=True)
        
        df = pd.DataFrame()
        gdf = GeoDataFrame(df, geometry=geometry_predicted[i])
        world = gpd.read_file('geopandas_data/usa-states-census-2014.shp')
        fig = gdf.plot(
            ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15
        ).get_figure()
        fig.axes[0].axis("off")
        fig.savefig(os.path.join(output_path, f"{i}_predicted.png"), bbox_inches='tight', transparent=True)

    
if __name__ == "__main__":
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