import argparse
import yaml
import sys
import os
import time
import torch
import shutil
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
import torchvision

from sklearn.metrics import top_k_accuracy_score, f1_score

from ebird.model.model import Model
from ebird.model.checkpointer import Checkpointer
from ebird.datasets import build_dataset
from ebird.utils.utils import compute_multilabel_top_k_accuracy, compute_multilabel_special_top_k_accuracy

def run_one_epoch(
    config,
    epoch,
    model,
    optimizer,
    train_dataloader,
    validation_dataloader,
    criterion,
    writer,
    checkpoint_callback,
    device,
    iteration,
    is_train=True
):
    model.eval()

    running_loss = 0
    running_accuracy = 0
    outputs_acc = []
    targets_acc = []
    for i, (input_images, input_features, targets) in enumerate(validation_dataloader):
        outputs = model(input_images.to(device), input_features.to(device))

        outputs_acc.append(outputs.detach().cpu())
        targets_acc.append(targets.detach().cpu())

        if config["BOOST_LOSS"]:
            loss = criterion(outputs, targets.to(device))
            loss[targets == 1] *= 10
            loss = loss.mean()
        else:
            loss = criterion(outputs, targets.to(device))

        if len(targets.shape) == 1:
            accuracy = torch.sum(
                outputs.argmax(axis=1).detach().cpu() == targets
            ) / config["BATCH_SIZE"]
        else:
            accuracy = torch.sum(
                (torch.sigmoid(outputs.detach().cpu()) > 0.5) == targets
            ) / (config["BATCH_SIZE"] * targets.shape[1])

        running_loss += loss.item()
        running_accuracy += accuracy

    if len(targets.shape) == 1:
        print("Not implemented for single target")
    if len(targets.shape) == 2:
        f1 = f1_score(
            np.nan_to_num(torch.concat(targets_acc).numpy()),
            (np.nan_to_num(torch.nn.functional.softmax(torch.concat(outputs_acc), dim=-1).numpy()) > 0.5),
            labels=[i for i in range(outputs.shape[-1])],
            average='micro'
        )
        print(f"F1 score: {f1}")
        multi_label_top_k = compute_multilabel_top_k_accuracy(
            torch.concat(targets_acc).numpy(),
            torch.concat(outputs_acc).numpy()
        )
        print(f"Multi-label accuracy score: {multi_label_top_k}")
        multi_label_special_top_k = compute_multilabel_special_top_k_accuracy(
            torch.concat(targets_acc).numpy(),
            torch.concat(outputs_acc).numpy()
        )
        print(f"Multi-label special accuracy score: {multi_label_special_top_k}")

    return iteration + i

def main(config_path):
    with open(config_path, "r") as conf:
        try:
            config = yaml.safe_load(conf)
        except yaml.YAMLError as ex:
            print(ex)
            sys.exit()

    device = "cuda"

    model = Model(config["MODEL"]).to(device)

    if config["TRAINING"]["LOSS"] == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        if config["TRAINING"]["BOOST_LOSS"]:
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

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
                mean=[890.3597,  929.5649,  690.5411, 2812.1230],
                std=[884.0230,  748.8593,  750.9115, 1343.0872],
            ),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

    # Validation == Test for eval script
    validation_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["VALIDATION"], transform=transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        num_workers=16
    )

    checkpoint = torch.load(config["RESUME"])
    new_state_dict = {
        k:(v if v.size() == model.state_dict()[k].size() else model.state_dict()[k])
        for k, v in zip(model.state_dict().keys(), checkpoint["model_state_dict"].values())
    }
    model.load_state_dict(new_state_dict, strict=False)
    
    iteration = run_one_epoch(
        config["TRAINING"],
        0,
        model,
        None,
        None,
        validation_dataloader,
        criterion,
        None,
        None,
        device,
        0,
        is_train=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser("eBird evaluation script")
    parser.add_argument("-c", "--configuration", type=str, nargs="?", help="Path to your configuration file", required=True)
    args = parser.parse_args()
    main(args.configuration)