def run_one_epoch(
    config,
    model,
    optimizer,
    dataloader,
    criterion,
    writer,
    checkpoint_callback,
    evaluation_callback
):
    for inputs, targets in dataloader:
        output = model(inputs, targets)

        loss = criterion(inputs, targets)