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
from ebird.utils.utils import compute_multilabel_top_k_accuracy

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
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_accuracy = 0
    outputs_acc = []
    targets_acc = []
    label_count_acc = []
    for i, (input_images, input_features, targets) in enumerate(train_dataloader if is_train else validation_dataloader):
        outputs, label_count = model(input_images.to(device), input_features.to(device))

        outputs_acc.append(outputs.detach().cpu())
        label_count_acc.append(label_count.detach().cpu())
        targets_acc.append(targets.detach().cpu())

        if config["BOOST_LOSS"]:
            loss = criterion(outputs, targets.to(device))
            loss[targets == 1] *= 10
            loss = loss.mean()
        else:
            loss = criterion(outputs, targets.to(device))

        if label_count is not None:
            loss += 0.001 * torch.nn.functional.l1_loss(label_count.squeeze(), targets.to(device).sum(dim=-1).squeeze())

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

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if is_train and i % config["LOGGING_INTERVAL"] == 0 and i != 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {running_loss}, Accuracy: {running_accuracy / config['LOGGING_INTERVAL']}")
            running_loss = 0
            running_accuracy = 0

        if is_train and i % config["EVALUATION_INTERVAL"] == 0 and i != 0:
            # FIXME: Recursivity in this context is bad
            run_one_epoch(
                config,
                epoch,
                model,
                optimizer,
                None,
                validation_dataloader,
                criterion,
                writer,
                None,
                device,
                iteration + i,
                False
            )

        if is_train and i % config["CHECKPOINT_INTERVAL"] == 0 and i != 0:
            checkpoint_callback(model, optimizer, epoch, i, loss)

        writer.add_scalar(f"{'train' if is_train else 'eval'}/loss", loss.item(), iteration + i)
        if len(targets.shape) == 1:
            writer.add_scalar(f"{'train' if is_train else 'eval'}/top_1_accuracy",
                top_k_accuracy_score(
                    targets.detach().cpu().numpy(),
                    torch.nn.functional.softmax(outputs.detach().cpu(), dim=-1).numpy(),
                    k=1,
                    labels=[i for i in range(outputs.shape[-1])]),
                iteration + i
            )
            writer.add_scalar(f"{'train' if is_train else 'eval'}/top_5_accuracy",
                top_k_accuracy_score(
                    targets.detach().cpu().numpy(),
                    torch.nn.functional.softmax(outputs.detach().cpu(), dim=-1).numpy(),
                    k=5,
                    labels=[i for i in range(outputs.shape[-1])]),
                iteration + i
            )
            writer.add_scalar(f"{'train' if is_train else 'eval'}/top_30_accuracy",
                top_k_accuracy_score(
                    targets.detach().cpu().numpy(),
                    torch.nn.functional.softmax(outputs.detach().cpu(), dim=-1).numpy(),
                    k=30,
                    labels=[i for i in range(outputs.shape[-1])]),
                iteration + i
            )
        if len(targets.shape) == 2:
            writer.add_scalar(f"{'train' if is_train else 'eval'}/f1_score",
                f1_score(
                    np.nan_to_num(targets.detach().cpu().numpy()),
                    (np.nan_to_num(torch.nn.functional.softmax(outputs.detach().cpu(), dim=-1).numpy()) > 0.5),
                    labels=[i for i in range(outputs.shape[-1])],
                    average='micro'
                ),
                iteration + i
            )
            writer.add_scalar(f"{'train' if is_train else 'eval'}/multilabel_top_30_accuracy",
                compute_multilabel_top_k_accuracy(
                    targets.detach().cpu().numpy(),
                    torch.nn.functional.softmax(outputs.detach().cpu(), dim=-1).numpy()
                ),
                iteration + i
            )


    if len(targets.shape) == 1:
        writer.add_scalar(f"{'train' if is_train else 'eval'}/epoch_top_1_accuracy",
            top_k_accuracy_score(
                torch.concat(targets_acc).numpy(),
                torch.nn.functional.softmax(torch.concat(outputs_acc), dim=-1).numpy(),
                k=1,
                labels=[i for i in range(outputs.shape[-1])]),
            iteration + i
        )
        writer.add_scalar(f"{'train' if is_train else 'eval'}/epoch_top_5_accuracy",
            top_k_accuracy_score(
                torch.concat(targets_acc).numpy(),
                torch.nn.functional.softmax(torch.concat(outputs_acc), dim=-1).numpy(),
                k=5,
                labels=[i for i in range(outputs.shape[-1])]),
            iteration + i
        )
        writer.add_scalar(f"{'train' if is_train else 'eval'}/epoch_top_30_accuracy",
            top_k_accuracy_score(
                torch.concat(targets_acc).numpy(),
                torch.nn.functional.softmax(torch.concat(outputs_acc), dim=-1).numpy(),
                k=30,
                labels=[i for i in range(outputs.shape[-1])]),
            iteration + i
        )
    if len(targets.shape) == 2:
        writer.add_scalar(f"{'train' if is_train else 'eval'}/f1_score",
            f1_score(
                np.nan_to_num(torch.concat(targets_acc).numpy()),
                (np.nan_to_num(torch.nn.functional.softmax(torch.concat(outputs_acc), dim=-1).numpy()) > 0.5),
                labels=[i for i in range(outputs.shape[-1])],
                average='micro'
            ),
            iteration + i
        )
        writer.add_scalar(f"{'train' if is_train else 'eval'}/multilabel_top_30_accuracy",
            compute_multilabel_top_k_accuracy(
                torch.concat(targets_acc).numpy(),
                torch.concat(outputs_acc).numpy()
            ),
            iteration + i
        )
        writer.add_scalar(f"{'train' if is_train else 'eval'}/label_count_average_absolute_error",
            torch.mean(torch.abs(torch.cat(label_count_acc, dim=0).squeeze() - torch.cat(targets_acc, dim=0).sum(dim=-1))),
            iteration + i
        )

    if is_train:
        checkpoint_callback(model, optimizer, epoch, i, loss)

    if is_train:
        # FIXME: Recursivity in this context is bad
        run_one_epoch(
            config,
            epoch,
            model,
            optimizer,
            None,
            validation_dataloader,
            criterion,
            writer,
            None,
            device,
            iteration + i,
            False
        )

    return iteration + i

def main(config_path):
    with open(config_path, "r") as conf:
        try:
            config = yaml.safe_load(conf)
        except yaml.YAMLError as ex:
            print(ex)
            sys.exit()
    output_path = os.path.join(config["OUTPUT_DIR"], os.path.basename(config_path), str(time.time()))
    os.makedirs(output_path)
    shutil.copyfile(config_path, os.path.join(output_path, "config.yaml"))
    writer = SummaryWriter(log_dir=output_path)
    checkpointer = Checkpointer(output_path)

    device = "cuda"

    model = Model(config["MODEL"]).to(device)

    if config["TRAINING"]["OPTIMIZER"]["ALGORITHM"] == "Adam":
        optimizer = Adam(model.parameters(), lr=config["TRAINING"]["OPTIMIZER"]["LEARNING_RATE"])
    elif config["TRAINING"]["OPTIMIZER"]["ALGORITHM"] == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config["TRAINING"]["OPTIMIZER"]["LEARNING_RATE"])
    else:
        print("Unknown optimizer algorithm")

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
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[890.3597,  929.5649,  690.5411, 2812.1230],
                std=[884.0230,  748.8593,  750.9115, 1343.0872],
            ),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(degrees=(0, 360)),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])
        valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[890.3597,  929.5649,  690.5411, 2812.1230],
                std=[884.0230,  748.8593,  750.9115, 1343.0872],
            ),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ])

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["TRAIN"], transform=train_transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=16
    )
    validation_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["VALIDATION"], transform=valid_transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        num_workers=16
    )

    iteration = 0
    epoch = 0

    if "RESUME" in config and config["RESUME"]:
        checkpoint = torch.load(config["RESUME"])
        new_state_dict = {
            k:(v if v.size() == model.state_dict()[k].size() else model.state_dict()[k])
            for k, v in zip(model.state_dict().keys(), checkpoint["model_state_dict"].values())
        }
        model.load_state_dict(new_state_dict, strict=False)
        if not config.get("RESET_TRAINING", False):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            iteration = checkpoint["iteration"]

    for current_epoch in range(epoch, config["TRAINING"]["EPOCHS"]):
        iteration = run_one_epoch(
            config["TRAINING"],
            current_epoch,
            model,
            optimizer,
            train_dataloader,
            validation_dataloader,
            criterion,
            writer,
            checkpointer.save,
            device,
            iteration,
            is_train=True
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser("eBird training script")
    parser.add_argument("-c", "--configuration", type=str, nargs="?", help="Path to your configuration file", required=True)
    args = parser.parse_args()
    main(args.configuration)