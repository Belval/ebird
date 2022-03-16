import argparse
import yaml
import sys
import os
import time
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ebird.model.model import Model
from ebird.model.checkpointer import Checkpointer
from ebird.datasets import build_dataset

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
    for i, (inputs, targets) in enumerate(train_dataloader if is_train else validation_dataloader):
        outputs = model(inputs.to(device))

        loss = criterion(outputs, targets.to(device))
        if len(targets.shape) == 1:
            accuracy = torch.sum(
                outputs.argmax(axis=1).detach().cpu() == targets
            ) / config["BATCH_SIZE"]
        else:
            accuracy = torch.sum(
                (torch.nn.functional.sigmoid(outputs.detach().cpu()) > 0.5) == targets
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
        writer.add_scalar(f"{'train' if is_train else 'eval'}/accuracy", accuracy, iteration + i)
    return iteration + i

def main(config):
    output_path = os.path.join(config["OUTPUT_DIR"], str(time.time()))
    os.makedirs(output_path)
    writer = SummaryWriter(log_dir=output_path)
    checkpointer = Checkpointer(output_path)

    device = "cuda"

    model = Model(config["MODEL"]).to(device)
    optimizer = Adam(model.parameters(), lr=config["TRAINING"]["OPTIMIZER"]["LEARNING_RATE"])

    if config["TRAINING"]["LOSS"] == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[111.9393, 121.3099, 113.0863, 140.8277],
            std=[51.5302,  45.5618,  41.4096,  54.2996],
        ),
        torchvision.transforms.ConvertImageDtype(torch.float),
    ])

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["TRAIN"], transform=transform),
        batch_size=config["TRAINING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=16
    )
    validation_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["VALIDATION"], transform=transform),
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
        if not config["RESET_TRAINING"]:
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
    with open(args.configuration, "r") as conf:
        try:
            config = yaml.safe_load(conf)
        except yaml.YAMLError as ex:
            print(ex)
            sys.exit()
    main(config)