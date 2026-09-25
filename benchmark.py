import argparse
import gc
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from calpeft.config import Config
from calpeft.losses import total_loss
from calpeft.models import MODELS, build_model, count_parameters
from calpeft.utils import set_seed


def profile(cfg, batch=2, steps=10, warmup=3):
    set_seed(cfg.seed)
    model = build_model(cfg)
    total, trainable = count_parameters(model)
    cuda = cfg.device == "cuda"
    amp = (lambda: torch.amp.autocast("cuda")) if cuda else nullcontext
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    x = torch.rand(batch, 3, cfg.image_size, cfg.image_size, device=cfg.device)
    y = (torch.rand(batch, 1, cfg.image_size, cfg.image_size, device=cfg.device) > 0.5).float()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t_train = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        with amp():
            loss = total_loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            t_train.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
    model.eval()
    t_infer = []
    with torch.no_grad(), amp():
        for i in range(warmup + steps):
            t0 = time.perf_counter()
            model(x[:1])
            if cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                t_infer.append(time.perf_counter() - t0)
    return {"model": cfg.model, "params_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 3),
            "trainable_pct": round(100 * trainable / total, 2), "train_peak_mem_GB": round(peak, 2),
            "train_s_per_iter": round(float(np.median(t_train)), 3),
            "infer_ms_per_img": round(1000 * float(np.median(t_infer)), 1)}


def main():
    parser = argparse.ArgumentParser(description="Parameters, peak GPU memory and speed of every model (batch 2).")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    rows = []
    for name in args.models:
        try:
            rows.append(profile(Config(model=name, ckpt_dir=args.ckpt_dir)))
            print(rows[-1])
        except Exception as e:
            print(f"[FAILED] {name}: {type(e).__name__}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "tables", "efficiency.csv"), index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
