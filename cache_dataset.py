from atm.dataloader.cache_dataset import create_cached_augmented_dataset
from yaml import safe_load
import argparse

# Get args from cli
parser = argparse.ArgumentParser()
# Get the dataset bc or track
parser.add_argument("--dataset", default="bc", choices=["bc", "atm"], help="The name of the desired dataset.")
args = parser.parse_args()
dataset_name = args.dataset

cfg = safe_load(open(f'{dataset_name}_dataset_args.yaml'))

create_cached_augmented_dataset(dataset_name=dataset_name, **cfg)