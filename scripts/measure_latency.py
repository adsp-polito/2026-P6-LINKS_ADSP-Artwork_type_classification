"""
Single-image latency: time from the image to the predicted type label, on gpu and on cpu.
"""
import argparse
import io
import json
import os
import platform
import time
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image

from src.data import build_transform
from src.training import ArtworkClassifier

DEVICES = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
USE_AMP = True
NUM_IMAGES = 500
NUM_WARMUP = 20
STAGES = ["load", "preprocess", "inference", "total"]
WANDB_STAGES = ["inference", "total"] #log only inference and total latency to wandb
WANDB_STATS = ["median"] #log only median latency to wandb


def cpu_name():
    """ Returns CPU name """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown cpu"


def predict(model, transform, data, device):
    """Returns the predicted label and the time of each stage in ms, for one image given as bytes in memory.
    load = decode, preprocess = resize + to tensor + normalize, inference = to device + forward + argmax."""
    amp = USE_AMP and device == "cuda"
    t0 = time.perf_counter()
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
    t1 = time.perf_counter()
    x = transform(img).unsqueeze(0)
    t2 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=amp):
        logits = model(x.to(device))
    label = model.classes[logits.argmax(1).item()]  # .item() waits for the gpu to finish
    t3 = time.perf_counter()
    times = {"load": t1 - t0, "preprocess": t2 - t1, "inference": t3 - t2, "total": t3 - t0}
    return label, {k: v * 1000 for k, v in times.items()}


def measure(model, transform, images, device):
    model = model.to(device)
    for data in images[:NUM_WARMUP]:
        predict(model, transform, data, device)

    times = {k: [] for k in STAGES}
    for data in images[NUM_WARMUP:]:
        _, t = predict(model, transform, data, device)
        for k in STAGES:
            times[k].append(t[k])

    n = len(times["total"])
    amp = USE_AMP and device == "cuda"
    if device == "cuda":
        name = torch.cuda.get_device_name()
        info = {"gpu": name}
    else:
        name = f"{cpu_name()} ({os.cpu_count()} cores, {torch.get_num_threads()} torch threads)"
        info = {"cpu": name}
    print(f"\n{name} | amp {amp} | {n} images")
    summary = {}
    for k in STAGES:
        v = np.array(times[k])
        summary[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                      "p95": float(np.percentile(v, 95)), "max": float(v.max())}
        print(f"{k:>10} ms | mean {v.mean():8.2f} | median {np.median(v):8.2f} | "
              f"p95 {np.percentile(v, 95):8.2f} | max {v.max():8.2f}")
    return {**info, "amp": amp, "num_images": n, "ms": summary}


def log_to_wandb(results, json_path, project, entity, run_name):
    """Log latency results to wandb: only the times, device info is in latency.json"""
    summary = {}
    for device, r in results.items():
        for stage in WANDB_STAGES:
            for stat in WANDB_STATS:
                summary[f"latency_{device}/{stage}_{stat}_ms"] = r["ms"][stage][stat]
    run = wandb.init(project=project, entity=entity, id=run_name, resume="must")
    # log (not only summary): one value per run in the history, so wandb draws the bar chart panels by itself
    run.log(summary)
    run.save(str(json_path), base_path=str(json_path.parent), policy="now")
    run.finish()
    print(f"logged latency to wandb run {run_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="training config")
    parser.add_argument("--ckpt", default="best", choices=["best", "last"])
    parser.add_argument("--no_wandb", action="store_true", help="only save latency.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    run_name = config["run_name"]
    run_folder = Path(config.get("out_dir", "runs")) / run_name
    checkpoint = run_folder / "checkpoints" / f"{args.ckpt}.ckpt"

    model = ArtworkClassifier.load_from_checkpoint(checkpoint, map_location="cpu", pretrained=False).eval()
    hp = model.hparams
    cfg = timm.get_pretrained_cfg(hp.model_name)
    transform = build_transform(hp.img_size, cfg.mean, cfg.std)

    df = pd.read_csv(config["data"]["splits"])
    test = df[df["split"] == "test"]
    test = test.sample(n=min(NUM_WARMUP + NUM_IMAGES, len(test)), random_state=42)
    # read once to avoid measuring disk read time
    images = [(Path(config["data"]["data_root"]) / p).read_bytes() for p in test["images_path"]]

    print(f"{run_name} ({args.ckpt}.ckpt) | {hp.model_name} | img_size {hp.img_size}")
    results = {d: measure(model, transform, images, d) for d in DEVICES}

    out = run_folder / "latency.json"
    out.write_text(json.dumps({"model_name": hp.model_name, "img_size": hp.img_size, "checkpoint": args.ckpt,
                               **results}, indent=2))
    print(f"\nsaved {out}")

    if config.get("wandb_project") and not args.no_wandb:
        log_to_wandb(results, out, config["wandb_project"], config.get("wandb_entity"), run_name)


if __name__ == "__main__":
    main()
