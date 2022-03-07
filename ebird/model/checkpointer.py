import os
import torch

class Checkpointer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, model, optimizer, epoch, iteration, loss):
        torch.save(
            {
                "epoch": epoch,
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(self.output_dir, f"checkpoint_{epoch}_{iteration}.pth")
        )