import argparse
import os

from calpeft.config import Config
from calpeft.data import DATASETS
from calpeft.engine import run
from calpeft.models import MODELS


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate one model on one dataset.")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=None, help="default: the model's value in calpeft/models.py")
    parser.add_argument("--batch-size", type=int, default=None, help="default: the model's value in calpeft/models.py")
    parser.add_argument("--lambda-cal", type=float, default=0.1, help="weight of the calibration loss")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--tag", default="", help="added to the run name, e.g. s43 -> outputs/runs/<dataset>/<model>-s43")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--eval-only", action="store_true", help="evaluate an existing model.pt without training")
    parser.add_argument("--force", action="store_true", help="train again even if the run is finished")
    args = parser.parse_args()
    tag = "-" + args.tag if args.tag and not args.tag.startswith("-") else args.tag

    cfg = Config(dataset=args.dataset, model=args.model, seed=args.seed, epochs=args.epochs, lr=args.lr,
                 batch_size=args.batch_size, lambda_cal=args.lambda_cal, lora_rank=args.lora_rank, tag=tag,
                 data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, out_dir=args.out_dir)
    if os.path.exists(os.path.join(cfg.run_dir, "metrics.json")) and not (args.force or args.eval_only):
        print(f"{cfg.run_dir} is already finished (use --force to train again)")
        return
    run(cfg, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
