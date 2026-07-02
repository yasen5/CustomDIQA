"""Populate a checkpoint's optimizer state from scratch.

Older checkpoints were saved before AdamW state (`exp_avg` / `exp_avg_sq`)
was persisted alongside the weights (see src/trainer.py). Resuming training
from one of those with a fresh optimizer makes Adam's bias-corrected update
for the first several steps come out ~lr * sign(grad) for every parameter —
a near-uniform per-parameter step regardless of true gradient size — which
is disruptive to an already-converged model (loss spikes, see git history
for context).

This script runs a handful of *real* optimizer steps (real data, real
gradients, real weight updates) purely to populate Adam's moment buffers,
then restores the checkpoint's original weights before saving — so the
saved weights are untouched, but the optimizer now starts "warm" instead
of cold on the next resume.
"""
import argparse
import copy
import os
import random
import sys
import types
from datetime import datetime
from time import perf_counter

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from src.constants import (
    TRAIN_BACKBONE_LR_SCALE_DEFAULT,
    TRAIN_BATCH_SIZE_DEFAULT,
    TRAIN_GRAD_ACCUM_DEFAULT,
    TRAIN_HEAD_WEIGHT_DECAY_DEFAULT,
    TRAIN_LR_DEFAULT,
    TRAIN_SAMPLE_SEED_DEFAULT,
    TRAIN_WEIGHT_DECAY_DEFAULT,
)
from src.datasets.single_dataset import SingleDataset
from src.model import build_model, load_checkpoint, load_model_type
from src.trainer import SimpleImageProcessor, collate, get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint dir containing weights.pt")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--lr", type=float, default=TRAIN_LR_DEFAULT)
    parser.add_argument("--backbone-lr-scale", type=float, default=TRAIN_BACKBONE_LR_SCALE_DEFAULT)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--head-weight-decay", type=float, default=TRAIN_HEAD_WEIGHT_DECAY_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE_DEFAULT)
    parser.add_argument("--grad-accum", type=int, default=TRAIN_GRAD_ACCUM_DEFAULT)
    parser.add_argument("--burn-in-steps", type=int, default=20,
                        help="Number of real optimizer steps used to populate Adam moments")
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SAMPLE_SEED_DEFAULT)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_type = load_model_type(args.checkpoint_path)
    model, model_constants = build_model(model_type)
    model = model.to(device=device, dtype=torch.float32)

    weights_path = os.path.join(args.checkpoint_path, "weights.pt")
    model_state, optimizer_state = load_checkpoint(weights_path, map_location=device)
    if optimizer_state is not None:
        raise ValueError(f"{weights_path} already has optimizer state — nothing to warm-start")
    model.load_state_dict(model_state)
    original_state = copy.deepcopy(model.state_dict())
    print(f"Loaded {model_type} model from {weights_path}")

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.parameters() if not any(p is h for h in head_params)]
    target_lrs = [args.lr, args.lr * args.backbone_lr_scale]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 0.0, "weight_decay": args.head_weight_decay},
        {"params": backbone_params, "lr": 0.0, "weight_decay": args.weight_decay},
    ])
    print(f"  Head LR (post-resume): {target_lrs[0]:.1e} (wd {args.head_weight_decay})  "
          f"Backbone LR (post-resume): {target_lrs[1]:.1e} (scale {args.backbone_lr_scale}, wd {args.weight_decay})")
    print("  Burn-in itself runs at lr=0 for every group — Adam's exp_avg/exp_avg_sq/step still "
          "update on every optimizer.step() call (they aren't gated by lr), but the parameter update "
          "(scaled by lr) is exactly zero, so weights never move and every burn-in gradient is sampled "
          "at the true, unperturbed checkpoint minimum instead of a self-perturbed trajectory.")

    processor = SimpleImageProcessor(model_constants.img_size)
    data_args = types.SimpleNamespace(
        data_paths=[args.data_path],
        data_weights=[1],
        image_folder=args.image_folder,
        image_processor=processor,
        image_aspect_ratio="pad",
    )
    dataset = SingleDataset(data_paths=data_args.data_paths, data_weights=data_args.data_weights, data_args=data_args)
    pool_indices = list(range(len(dataset)))
    print(f"Dataset: {len(pool_indices)} samples from {args.data_path}")

    model.train()
    random.seed(args.sample_seed)
    print(f"\nRunning {args.burn_in_steps} real optimizer steps "
          f"(batch {args.batch_size}, grad-accum {args.grad_accum}) to populate Adam moments...")
    burn_in_started_at = perf_counter()
    for step in range(1, args.burn_in_steps + 1):
        step_started_at = perf_counter()
        step_loss = 0.0
        for _ in range(args.grad_accum):
            indices = random.sample(pool_indices, args.batch_size)
            batch = collate([dataset[i] for i in indices])
            images = batch["images"].to(device=device, dtype=torch.float32)
            level_probs = batch["level_probs"].to(device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), level_probs, reduction="batchmean") / args.grad_accum
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"  burn-in step {step}/{args.burn_in_steps}  loss {step_loss:.6f}", flush=True)
        completed_at = datetime.now().isoformat(timespec="seconds")
        step_elapsed = perf_counter() - step_started_at
        total_elapsed = perf_counter() - burn_in_started_at
        print(f"  [perf] burn-in step {step}/{args.burn_in_steps} completed_at {completed_at}"
              f"  step_sec {step_elapsed:.3f}  total_sec {total_elapsed:.3f}", flush=True)

    # lr=0 means weights never moved — restore is just a safety net against
    # any float drift, and doubles as an assertion that nothing else changed them.
    max_drift = max((p1 - p2).abs().max().item() for p1, p2 in
                     zip(model.state_dict().values(), original_state.values()))
    print(f"\nMax weight drift during burn-in: {max_drift:.2e} (should be ~0, confirms lr=0 held)")
    model.load_state_dict(original_state)

    for pg, lr in zip(optimizer.param_groups, target_lrs):
        pg["lr"] = lr

    backup_path = weights_path + ".bak"
    if not os.path.exists(backup_path):
        os.rename(weights_path, backup_path)
        print(f"\nBacked up original weights to {backup_path}")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, weights_path)
    print(f"Saved warm-started checkpoint (original weights + burned-in optimizer state) to {weights_path}")
