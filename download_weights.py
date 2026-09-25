import argparse
import os

from calpeft.models import MODELS, ensure_weights
from calpeft.utils import sha256


def main():
    with_weights = [m for m, spec in MODELS.items() if spec["weights"]]
    parser = argparse.ArgumentParser(description="Download the pretrained weights used by the models.")
    parser.add_argument("--models", nargs="+", default=with_weights, choices=with_weights)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    args = parser.parse_args()

    failed = []
    for name in args.models:
        try:
            path = ensure_weights(name, args.ckpt_dir)
        except RuntimeError as e:
            print(e)
            failed.append(name)
            continue
        ok = sha256(path) == MODELS[name]["weights"][2]
        print(f"  {name:9s} {os.path.basename(path):30s} {os.path.getsize(path) / 2**20:6.0f} MB  "
              f"{'checksum ok' if ok else 'CHECKSUM MISMATCH - not the file used in the paper (delete it and run again)'}")
        if not ok:
            failed.append(name)
    print("\nall weights ready" if not failed else f"\nproblems with: {', '.join(failed)}")


if __name__ == "__main__":
    main()
