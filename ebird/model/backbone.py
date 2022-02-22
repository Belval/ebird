from torchvision.models import *

def get_backbone(config):
    return globals()[config["CLASS"]](pretrained=config["PRETRAINED"])