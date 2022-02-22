from ebird.datasets.geolifeclef import GeoLifeCLEFDataset
from ebird.datasets.ebird import EBirdDataset

def build_dataset(config):
    return globals()[config["TYPE"]](config)