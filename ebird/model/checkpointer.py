import os
import torch

class Checkpointer:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, model, epoch, iteration):
        torch.save(model.state_dict(), os.path.join(self.output_dir, f"checkpoint_{epoch}_{iteration}.pth"))