from ebird.datasets.geolifeclef import GeoLifeCLEFDataset
from ebird.datasets.ebird import EBirdDataset

def build_dataset(config, transform=None):
    return globals()[config["TYPE"]](config, transform=transform)