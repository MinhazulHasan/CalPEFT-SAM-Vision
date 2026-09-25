import argparse
import os

from calpeft.config import Config
from calpeft.data import DATASETS, check_split, ensure_dataset, split_file, split_indices, write_split


def main():
    parser = argparse.ArgumentParser(description="Download / extract / check the datasets and save the fixed data split.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    for name in args.datasets:
        cfg = Config(dataset=name, data_dir=args.data_dir)
        try:
            images, masks = ensure_dataset(cfg)
        except RuntimeError as e:
            print(f"\n{e}\n")
            continue
        sizes = {k: len(v) for k, v in split_indices(len(images), cfg).items()}
        if os.path.exists(split_file(name)):
            same = check_split(cfg, images, masks)
            print(f"  split {sizes} {'matches' if same else 'DOES NOT match'} {split_file(name)}")
        else:
            print(f"  split {sizes} saved to {write_split(cfg, images, masks)}")


if __name__ == "__main__":
    main()
