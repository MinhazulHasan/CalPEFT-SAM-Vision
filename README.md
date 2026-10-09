# CalPEFT-SAM

**Calibrated, parameter-efficient, conformally-guaranteed medical image segmentation on a single 8 GB GPU**

CalPEFT-SAM segments lesions (polyps, breast tumours) with a frozen SAM2.1 image encoder, small trainable LoRA adapters and a light decoder. It is trained with a calibration loss so that its confidence is meaningful, and it uses Conformal Risk Control (CRC) to pick a threshold with a statistical guarantee: on average the guaranteed region misses at most 10% of the true lesion pixels (FNR ≤ 0.10). Everything runs on one 8 GB laptop GPU.

This repository contains all the code needed to reproduce the experiments and results of the paper, starting from an empty computer.

## Results at a glance

Mean ± standard deviation over 3 training seeds (42, 43, 44). CRC values are the mean over 1000 random calibration/test re-splits (target FNR ≤ 0.10).

| Dataset | Model | Trained parameters | Dice ↑ | ECE ↓ | ACE ↓ | CRC FNR | CRC region size ↓ |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG | U-Net ResNet-34, full fine-tuning | 24.44 M | 0.8989 ± 0.0061 | 0.0185 ± 0.0026 | 0.1183 ± 0.0079 | 0.0939 ± 0.0003 | 0.1540 ± 0.0005 |
| Kvasir-SEG | CalPEFT-SAM (SAM2.1 + LoRA) | 0.57 M | 0.9115 ± 0.0045 | 0.0156 ± 0.0024 | 0.0926 ± 0.0214 | 0.0937 ± 0.0002 | 0.1488 ± 0.0011 |
| BUSI | U-Net ResNet-34, full fine-tuning | 24.44 M | 0.7464 ± 0.0110 | 0.0243 ± 0.0031 | 0.1689 ± 0.0109 | 0.0903 ± 0.0025 | 0.1123 ± 0.0050 |
| BUSI | CalPEFT-SAM (SAM2.1 + LoRA) | 0.57 M | 0.7641 ± 0.0065 | 0.0226 ± 0.0011 | 0.1299 ± 0.0134 | 0.0908 ± 0.0015 | 0.1126 ± 0.0056 |

The results of all 7 models are in [Models and expected results](#models-and-expected-results).

## What is in this repository

```
CalPEFT-SAM/
├── README.md               this guide
├── requirements.txt        Python packages (PyTorch is installed separately, see Step 3)
├── requirements-lock.txt   the exact package versions of the environment we used
├── LICENSE                 MIT license of this code
├── splits/                 the fixed train / calibration / test split of each dataset
├── calpeft/                the method itself
│   ├── config.py           all settings of one run (dataset, model, seed, epochs, ...)
│   ├── data.py             finding the images, the data split, resizing and augmentation
│   ├── models.py           U-Net baselines, frozen encoders, LoRA, decoder, pretrained weights
│   ├── losses.py           BCE + Dice + mL1-ACE calibration loss
│   ├── conformal.py        Conformal Risk Control (CRC) and the re-split test
│   ├── metrics.py          Dice, IoU, ECE, ACE, ROI-ECE, figures
│   └── engine.py           training, evaluation and saving of one run
├── prepare_data.py         Step 4: download / unzip / check the datasets
├── download_weights.py     Step 5: download the pretrained weights
├── train.py                train and evaluate one model on one dataset
├── run_experiments.py      Step 7: run a whole group of experiments of the paper
├── analyze.py              Step 8: make the tables, statistics and figures
└── benchmark.py            Step 8: parameters, GPU memory and speed of every model
```

Three folders are created during use and are not part of the repository (they are too large): `data/` (datasets), `checkpoints/` (pretrained weights) and `outputs/` (trained models and results).

---

# Step-by-step: reproduce the task on a new computer

The commands below are entered in a terminal, one line at a time, pressing Enter after each line.

## Step 0 — What we need

- A computer with an **NVIDIA GPU**. We used an NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB of memory; the models need only a few GB (SAM ViT-B, the largest encoder, peaks at about 2.2 GB). A CPU alone also works, but it is far too slow for training.
- **Windows 10/11 or Linux.**
- About **10 GB of free disk space**.
- An internet connection.
- Time: all experiments together take about **12 hours** of GPU time on our laptop. The runs can be stopped at any time and continued later.

## Step 1 — Install three programs

1. **NVIDIA driver** — download from https://www.nvidia.com/Download/index.aspx and install it. Check it: open a terminal and type `nvidia-smi`. A table with the name of the GPU should appear.
2. **Miniforge** (Python and the `conda` package manager, free for everyone) — https://github.com/conda-forge/miniforge. Download the installer for the operating system (Windows: `Miniforge3-Windows-x86_64.exe`) and install it with the default options. An existing Anaconda or Miniconda installation can be used instead; the commands are the same.
3. **Git** (downloads code) — https://git-scm.com/downloads. Install it with the default options.

Which terminal to use:
- **Windows:** open the Start menu and start **"Miniforge Prompt"**. Use this window for everything below.
- **Linux:** open a normal terminal.

## Step 2 — Download the code

```
git clone https://github.com/MinhazulHasan/CalPEFT-SAM-Vision.git
cd CalPEFT-SAM-Vision
```

## Step 3 — Create the Python environment

An environment is a separate box of Python packages just for this project.

```
conda create -n calpeft python=3.11 -y
conda activate calpeft
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The third line installs PyTorch with GPU (CUDA 12.8) support, which works for current NVIDIA GPUs including the RTX 50 series. For a different setup, https://pytorch.org/get-started/locally/ shows the right command. The last line also downloads SAM 2 from Meta's GitHub (this is why Git is needed) and can take a few minutes. For the paper we used a nightly build of PyTorch (`pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`); the code does not need any nightly-only feature, so the stable command above is recommended.

## Step 4 — Get the datasets

```
python prepare_data.py
```

- **Kvasir-SEG** (1000 colonoscopy images with polyp masks, 46 MB) is downloaded and unzipped automatically.
- **BUSI** (780 breast ultrasound images with tumour masks) must be downloaded by hand once, because Kaggle needs a free account:
  1. Open https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset, sign in and click **Download**. This gives a file called `archive.zip`.
  2. Move `archive.zip` into the `data` folder inside `CalPEFT-SAM` (do not unzip it manually).
  3. Run `python prepare_data.py` again. It unzips the file and checks it.

When everything is fine, the last lines are:

```
kvasir: found 1000 image/mask pairs
  split {'train': 700, 'calib': 150, 'test': 150} matches .../splits/kvasir.csv
busi: found 780 image/mask pairs
  split {'train': 546, 'calib': 117, 'test': 117} matches .../splits/busi.csv
```

"matches" means the training, calibration and test images are exactly the same as ours.

`data` folder now looks like this:

```
data/
├── Kvasir-SEG/
│   ├── images/        1000 .jpg images
│   └── masks/         1000 .jpg masks
└── Dataset_BUSI_with_GT/
    ├── benign/        images "benign (1).png" and masks "benign (1)_mask.png"
    ├── malignant/
    └── normal/
```

**What about preprocessing?** No preprocessing needs to be run by hand. It happens automatically every time an image is loaded:
- each image is resized to 256 × 256 pixels and its values are scaled to 0–1;
- each mask is resized to 256 × 256 and turned into 0 (background) or 1 (lesion);
- training images only are randomly flipped, rotated by 90°/180°/270° and changed a little in brightness and contrast (data augmentation);
- the images are divided into 70% training, 15% calibration (used only by CRC) and 15% test. This split is always the same (it is made with the fixed seed 42) and is saved in `splits/`.

## Step 5 — Download the pretrained weights

```
python download_weights.py
```

This downloads 6 files (about 1.2 GB) into the `checkpoints` folder and checks that each file is correct ("checksum ok").

| File | Pretrained model | Size | Source |
|---|---|---|---|
| `sam2.1_hiera_tiny.pt` | SAM 2.1 Hiera-Tiny | 149 MB | Meta (dl.fbaipublicfiles.com) |
| `MedSAM2_latest.pt` | MedSAM2 | 149 MB | huggingface.co/wanglab/MedSAM2 |
| `sam_vit_b_01ec64.pth` | SAM ViT-B | 358 MB | Meta (dl.fbaipublicfiles.com) |
| `medsam_vit_b_hf.bin` | MedSAM ViT-B | 358 MB | huggingface.co/wanglab/medsam-vit-base |
| `dinov2_vits14_pretrain.pth` | DINOv2 ViT-S/14 | 84 MB | Meta (dl.fbaipublicfiles.com) |
| `resnet34-b627a593.pth` | ResNet-34 (ImageNet) | 83 MB | torchvision (download.pytorch.org) |

If a download fails (for example because of a company or university firewall), the script prints the link. Open the link in a browser, save the file into the `checkpoints` folder with exactly the file name shown, and run `python download_weights.py` again.

## Step 6 — Quick test (about 2 minutes)

Before starting the long experiments, check that everything works with a 1-epoch run:

```
python train.py --dataset kvasir --model unet --epochs 1 --tag test
```

It ends with a line like `Dice=... IoU=... ECE=... -> outputs/runs/kvasir/unet-test`. The numbers are low because it trained for only 1 epoch; that is fine. Delete the folder `outputs/runs/kvasir/unet-test` afterwards.

## Step 7 — Run the experiments

The experiments are divided into four groups. Run them in this order:

```
python run_experiments.py main
python run_experiments.py ablation
python run_experiments.py encoders
python run_experiments.py seeds
```

| Group | What it trains | Runs | Time on our GPU |
|---|---|---|---|
| `main` | U-Net (from scratch), U-Net ResNet-34 and CalPEFT-SAM on both datasets | 6 | about 2 h |
| `ablation` | CalPEFT-SAM with calibration-loss weight 0 / 0.5 / 1.0 and with LoRA rank 2 / 4 / 16 (Kvasir-SEG) | 6 | about 3 h |
| `encoders` | the same recipe with other frozen encoders: MedSAM2, DINOv2, SAM, MedSAM (both datasets) | 8 | about 5.5 h |
| `seeds` | U-Net ResNet-34 and CalPEFT-SAM again with training seeds 43 and 44 (both datasets) | 8 | about 3 h |

`python run_experiments.py all` runs all four groups one after another (about 13 hours).

Good to know:
- **Runs can be stopped at any time** (close the window or press Ctrl+C). When the same command is run again, finished runs are skipped and the unfinished run starts again from its beginning.
- `python run_experiments.py all --list` shows which runs are finished (`[done]`) and which are not (`[todo]`).
- Each run prints its loss every 5 epochs and, at the end, its Dice, ECE and CRC results.

## Step 8 — Make the tables and figures

```
python analyze.py
python benchmark.py
```

`analyze.py` reads all finished runs and writes these files (it works with any subset of finished runs, too):

| File in `outputs/` | What it contains |
|---|---|
| `tables/main_comparison.csv` | CalPEFT-SAM vs U-Net ResNet-34, mean ± std over 3 seeds (the table at the top of this page) |
| `tables/main_statistics.csv` | the same comparison per test image: paired Wilcoxon test, paired t-test, bootstrap 95% CI |
| `tables/model_comparison.csv` | all 7 models (seed 42) with the paired Wilcoxon test against SAM2 (Holm-corrected) |
| `tables/crc_validity.csv` | CRC over 1000 random calibration/test re-splits: mean / sd / 95th percentile of the FNR, how often FNR > 0.10, region size |
| `tables/ablation.csv` | calibration-loss weight 0 / 0.1 / 0.5 / 1.0 and LoRA rank 2 / 4 / 8 / 16, including ROI-ECE |
| `tables/risk_efficiency.csv` | FNR and region size at the risk levels 0.05, 0.10, 0.15 and 0.20 |
| `tables/all_runs.csv` | one line per run with all its metrics |
| `figures/crc_resplit_fnr.png` | distribution of the test FNR over the 1000 re-splits for every model |
| `figures/model_comparison.png` | Dice, ECE and CRC region size of all 7 models on both datasets |

`benchmark.py` writes `outputs/tables/efficiency.csv`: total and trained parameters, peak GPU memory during training and time per training step and per test image (batch size 2, same GPU for every model).

## Step 9 — Where to find everything

```
outputs/
├── runs/
│   └── <dataset>/<run>/            for example runs/kvasir/sam2 or runs/busi/unet-r34-s43
│       ├── model.pt                the trained model
│       ├── metrics.json            Dice, IoU, ECE, ACE, ROI-ECE, CRC threshold, test FNR, region size, parameters
│       ├── per_image_dice.csv      Dice and IoU of every test image
│       ├── crc_curves.npz          per-image FNR and region size at 200 thresholds (used by analyze.py)
│       ├── reliability_diagram.png calibration plot
│       ├── predictions/            6 test images: ground truth, predicted probability, guaranteed region
│       └── config.json             every setting of this run
├── tables/                         made by analyze.py and benchmark.py
└── figures/                        made by analyze.py
```

Run names: `sam2` = seed 42, `sam2-s43` = seed 43, `sam2-nocal` = no calibration loss, `sam2-r4` = LoRA rank 4, and so on.

---

# Train a single model

`train.py` trains and evaluates one model on one dataset:

```
python train.py --dataset kvasir --model sam2
python train.py --dataset busi --model unet-r34 --seed 43 --tag s43
python train.py --dataset kvasir --model sam2 --lora-rank 4 --tag r4
python train.py --dataset kvasir --model sam2 --eval-only
```

| Option | Meaning | Default |
|---|---|---|
| `--dataset` | `kvasir` or `busi` | required |
| `--model` | `unet`, `unet-r34`, `sam2`, `medsam2`, `sam1`, `medsam`, `dinov2s` | required |
| `--seed` | training seed (the data split stays the same) | 42 |
| `--epochs` | number of training epochs | 40 |
| `--lr` | learning rate | the model's value in the table below |
| `--batch-size` | batch size | the model's value in the table below |
| `--lambda-cal` | weight of the calibration loss (0 switches it off) | 0.1 |
| `--lora-rank` | LoRA rank | 8 |
| `--tag` | text added to the run name, e.g. `--tag s43` → `outputs/runs/<dataset>/sam2-s43` | none |
| `--eval-only` | do not train; evaluate the `model.pt` that is already in the run folder | off |
| `--force` | train again even if the run is finished | off |
| `--data-dir`, `--ckpt-dir`, `--out-dir` | other folders for data, pretrained weights, results | `data`, `checkpoints`, `outputs` |

`--eval-only` also evaluates a trained `model.pt` obtained from someone else: put it at `outputs/runs/<dataset>/<run>/model.pt` and run the command with the same `--dataset`, `--model` (and `--tag`).

# Models and expected results

All foundation encoders use exactly the same recipe (frozen encoder + LoRA rank 8 + the same decoder, loss, data and CRC); only the encoder changes. Dice below is from our seed-42 runs.

| Model | `--model` | Pretrained on | Trained / total parameters | lr | batch | Dice Kvasir-SEG | Dice BUSI |
|---|---|---|---|---|---|---|---|
| U-Net from scratch | `unet` | nothing | 1.93 M / 1.93 M | 1e-3 | 8 | 0.6432 | 0.5729 |
| U-Net ResNet-34, full fine-tuning | `unet-r34` | ImageNet | 24.44 M / 24.44 M | 1e-4 | 8 | 0.8958 | 0.7578 |
| DINOv2 ViT-S/14 + LoRA | `dinov2s` | natural images (self-supervised) | 0.566 M / 22.62 M | 1e-4 | 2 | 0.9044 | 0.7863 |
| SAM ViT-B + LoRA | `sam1` | natural images (SA-1B) | 0.837 M / 90.51 M | 1e-3 | 2 | 0.9063 | 0.7612 |
| MedSAM ViT-B + LoRA | `medsam` | medical images | 0.837 M / 90.51 M | 1e-3 | 2 | 0.9085 | 0.7334 |
| **CalPEFT-SAM:** SAM 2.1 Hiera-T + LoRA | `sam2` | natural images and videos (SA-1B, SA-V) | 0.565 M / 27.78 M | 1e-3 | 2 | 0.9088 | 0.7654 |
| MedSAM2 Hiera-T + LoRA | `medsam2` | medical images and videos | 0.565 M / 27.78 M | 1e-3 | 2 | 0.9118 | 0.7380 |

DINOv2 uses a learning rate of 1e-4 because its training diverged at 1e-3.

# Method in short

- **Encoder:** the pretrained image encoder is frozen. The 256 × 256 input is resized to 512 × 512 for the encoder (518 × 518 for DINOv2, whose patch size is 14). SAM2 gives features at several scales; for ViT encoders the outputs of blocks 3, 6, 9 and 12 are used.
- **LoRA:** every attention linear layer (qkv / q / v / kv / proj) gets a trainable low-rank update W x + (α / r) · B A x with rank r = 8 and α = 16. B starts at zero, so training starts from the pretrained encoder.
- **Decoder:** a small feature-pyramid decoder (128 channels, GroupNorm) that fuses all feature scales and predicts one probability map at 256 × 256.
- **Loss:** binary cross-entropy + Dice + 0.1 × mL1-ACE, a differentiable calibration loss (Barfoot et al., IEEE TMI 2025).
- **Training:** AdamW, cosine learning-rate schedule, 40 epochs, mixed precision on the GPU.
- **Conformal Risk Control:** on the calibration images we choose the largest threshold t (from 200 values between 0.99 and 0) with (n / (n + 1)) · FNR_calibration(t) + 1 / (n + 1) ≤ α, where α = 0.10 and FNR is the fraction of lesion pixels that are missed. All pixels with probability ≥ t form the guaranteed region. This guarantees that the expected FNR on new images is at most α (Angelopoulos et al., ICLR 2024).
- **Evaluation:** Dice and IoU at threshold 0.5 (an empty prediction on an empty mask counts as Dice 1), pixel-level ECE and ACE with 15 bins, ROI-ECE (ECE only inside the true or predicted lesion), and CRC checked over 1000 random re-splits of the calibration + test images.

# Reproducibility notes

- **Same split for everyone.** The split is made with seed 42 and saved in `splits/`. `prepare_data.py` and every training run check that the local split matches it and print a warning if not.
- **Small differences are normal.** Training on a GPU is not exactly repeatable, even with the same seed. Reproduced numbers can differ from ours by about as much as the seed-to-seed spread in the table above (around ±0.01 Dice).
- **Same code as in the paper.** This repository is a cleaned-up version of the notebooks we used for the paper. We checked that, on the CPU, both give bit-for-bit identical trained weights and metrics for all 7 models.
- **BUSI details.** A few BUSI images have more than one mask file (`..._mask_1.png`, ...); we use only the first mask (`..._mask.png`). The 133 "normal" images have empty masks. BUSI is used as distributed; it is known to contain duplicate images (Pawłowska et al., Data in Brief 2023), which we did not remove.
- **Tested with** Python 3.11, PyTorch 2.12.0.dev20260408 (nightly build, CUDA 12.8), on Windows with an NVIDIA RTX 5050 Laptop GPU (8 GB).

# License

The code in this repository is released under the MIT license (see `LICENSE`). The datasets and the pretrained weights are **not** part of this repository and keep their own licenses and terms of use; please read them on the original pages (for example, Kvasir-SEG may be used for research and education only): [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/), [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), [SAM 2](https://github.com/facebookresearch/sam2), [SAM](https://github.com/facebookresearch/segment-anything), [MedSAM](https://github.com/bowang-lab/MedSAM), [MedSAM2](https://github.com/bowang-lab/MedSAM2), [DINOv2](https://github.com/facebookresearch/dinov2), [torchvision](https://github.com/pytorch/vision).

Questions and problems: please open an issue on the GitHub page of this repository.
