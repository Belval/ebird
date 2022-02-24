import torch
from torchvision.models import *

def get_backbone(config):
    backbone = globals()[config["CLASS"]](pretrained=config["PRETRAINED"])
    if config["INPUT_CHANNELS"] == 3:
        return backbone
    else:
        backbone.conv1 = torch.nn.Conv2d(config["INPUT_CHANNELS"], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return backbone
    