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
        accuracy = torch.sum(outputs.argmax(axis=1).detach().cpu() == targets) / config["BATCH_SIZE"]
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
            checkpoint_callback(model, epoch, i)

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
    criterion = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[111.9393, 121.3099, 113.0863, 140.8277,  24.5900, 318.8939],
            std=[ 51.5302,  45.5618,  41.4096,  54.2996,   4.3272, 495.6868],
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
    for epoch in range(config["TRAINING"]["EPOCHS"]):
        iteration = run_one_epoch(
            config["TRAINING"],
            epoch,
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