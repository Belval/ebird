import argparse
import yaml
import sys
import os
import time
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
    is_train=True
):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0
    for i, (inputs, targets) in enumerate(train_dataloader if is_train else validation_dataloader):
        outputs = model(inputs, targets)

        loss = criterion(outputs, targets)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_train and i % config["LOGGING_INTERVAL"] == 0 and i != 0:
            print(running_loss)

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
                False
            )

        if is_train and i % config["CHECKPOINT_INTERVAL"] == 0 and i != 0:
            checkpoint_callback(model, epoch, i)

        writer.add_scalar(f"{'train' if is_train else 'eval'}/loss", loss.item())

def main(config):
    output_path = os.path.join(config["OUTPUT_DIR"], str(time.time()))
    os.makedirs(output_path)
    writer = SummaryWriter(log_dir=output_path)
    checkpointer = Checkpointer(output_path)

    model = Model(config["MODEL"])
    optimizer = Adam(model.parameters(), lr=config["TRAINING"]["OPTIMIZER"]["LEARNING_RATE"])
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["TRAIN"]),
        batch_size=config["TRAINING"]["BATCH_SIZE"]
    )
    validation_dataloader = torch.utils.data.DataLoader(
        build_dataset(config["DATASET"]["VALIDATION"]),
        batch_size=config["TRAINING"]["BATCH_SIZE"]
    )

    for epoch in range(config["TRAINING"]["EPOCHS"]):
        run_one_epoch(
            config["TRAINING"],
            epoch,
            model,
            optimizer,
            train_dataloader,
            validation_dataloader,
            criterion,
            writer,
            checkpointer.save,
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