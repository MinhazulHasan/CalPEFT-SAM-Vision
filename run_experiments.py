import argparse
import gc
import os
import time

import torch

from calpeft.config import Config
from calpeft.engine import run

DATASETS = ["kvasir", "busi"]
EXPERIMENTS = {
    "main": [(ds, dict(model=m)) for ds in DATASETS for m in ("unet", "unet-r34", "sam2")],
    "ablation": [("kvasir", dict(model="sam2", lambda_cal=0.0, tag="-nocal"))]
                + [("kvasir", dict(model="sam2", lora_rank=r, tag=f"-r{r}")) for r in (2, 4, 16)]
                + [("kvasir", dict(model="sam2", lambda_cal=lam, tag=tag)) for lam, tag in ((0.5, "-cal05"), (1.0, "-cal10"))],
    "encoders": [(ds, dict(model=m)) for ds in DATASETS for m in ("medsam2", "dinov2s", "sam1", "medsam")],
    "seeds": [(ds, dict(model=m, seed=s, tag=f"-s{s}")) for s in (43, 44) for ds in DATASETS for m in ("unet-r34", "sam2")],
}
EXPERIMENTS["all"] = EXPERIMENTS["main"] + EXPERIMENTS["ablation"] + EXPERIMENTS["encoders"] + EXPERIMENTS["seeds"]


def main():
    parser = argparse.ArgumentParser(description="Run a group of experiments of the paper (finished runs are skipped).")
    parser.add_argument("experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--list", action="store_true", help="only show the runs and whether they are finished")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    start = time.time()
    for dataset, overrides in EXPERIMENTS[args.experiment]:
        cfg = Config(dataset=dataset, epochs=args.epochs, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                     out_dir=args.out_dir, **overrides)
        done = os.path.exists(os.path.join(cfg.run_dir, "metrics.json"))
        if args.list or done:
            print(f"{'[done]' if done else '[todo]'} {dataset}/{cfg.run_name}")
            continue
        try:
            run(cfg)
        except Exception as e:
            print(f"[FAILED] {dataset}/{cfg.run_name}: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not args.list:
        print(f"\nfinished in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
