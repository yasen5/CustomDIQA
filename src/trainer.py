import os
import random
import shutil
import signal
import sys
import types
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.constants import ANALYSIS_RESULTS_DIR
from src.datasets.single_dataset import SingleDataset
from src.model import build_model, load_checkpoint, load_model_type, save_model_type


class SimpleImageProcessor:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def __init__(self, img_size, augment=False):
        self.augment = augment
        self.crop_size = {"height": img_size, "width": img_size}
        steps = []
        if augment:
            # Mild only: aggressive crop/color-jitter can corrupt or crop out
            # the very distortions (blur/noise/compression) IQA labels are about.
            steps.append(transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))))
            steps.append(transforms.RandomCrop((img_size, img_size)))
            steps.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            steps.append(transforms.Resize((img_size, img_size)))
        steps += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ]
        self._transform = transforms.Compose(steps)

    def preprocess(self, image, return_tensors="pt"):
        return {"pixel_values": self._transform(image).unsqueeze(0)}


def collate(items):
    images = torch.stack([item.image for item in items])
    level_probs = torch.tensor([item.level_probs for item in items], dtype=torch.float32)
    return {"images": images, "level_probs": level_probs}


class OscillationAwareLR:
    def __init__(self, optimizer, warmup_steps=5, factor=0.5, threshold=0.3,
                 cooldown=50, min_lr=1e-6, ema_alpha=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.ema_alpha = ema_alpha
        self._global_step = 0
        self._ema_loss = None
        self._ema_osc = 0.0
        self._steps_since_reduce = cooldown
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._backbone_group_idx = None
        self._backbone_warmup_start = 0
        self._backbone_warmup_steps = 0

    @property
    def ema_osc(self):
        return self._ema_osc

    def begin_backbone_warmup(self, step, group_idx, warmup_steps):
        """Re-warms only `group_idx`'s LR from 0 over `warmup_steps`, starting
        at `step` (e.g. right after unfreezing a previously-frozen backbone
        param group), and resets the oscillation EMA so the legitimate
        one-time loss shift at unfreeze isn't read as instability."""
        self._backbone_group_idx = group_idx
        self._backbone_warmup_start = step
        self._backbone_warmup_steps = warmup_steps
        self._ema_loss = None
        self._ema_osc = 0.0
        self._steps_since_reduce = self.cooldown

    def step(self, loss):
        self._global_step += 1
        if self._global_step <= self.warmup_steps:
            scale = self._global_step / max(1, self.warmup_steps)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
            return
        if self._backbone_group_idx is not None:
            elapsed = self._global_step - self._backbone_warmup_start
            if elapsed <= self._backbone_warmup_steps:
                scale = elapsed / max(1, self._backbone_warmup_steps)
                pg = self.optimizer.param_groups[self._backbone_group_idx]
                pg["lr"] = self._base_lrs[self._backbone_group_idx] * scale
                return
            self._backbone_group_idx = None
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = (1 - self.ema_alpha) * self._ema_loss + self.ema_alpha * loss
        rel_dev = max(0.0, loss - self._ema_loss) / (self._ema_loss + 1e-8)
        self._ema_osc = (1 - self.ema_alpha) * self._ema_osc + self.ema_alpha * rel_dev
        self._steps_since_reduce += 1
        if self._ema_osc > self.threshold and self._steps_since_reduce >= self.cooldown:
            for pg in self.optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * self.factor, self.min_lr)
            print(f"  [osc-lr] step {self._global_step}: LR → {self.get_last_lr()[0]:.2e}"
                  f"  (osc index {self._ema_osc:.3f} > threshold {self.threshold})")
            self._steps_since_reduce = 0
            self._ema_osc = 0.0

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Device: {device}")

    model, model_constants = build_model(args.model_type, pretrained=args.pretrained)
    model = model.to(device=device, dtype=torch.float32)
    optimizer_state = None
    if args.checkpoint_path is not None:
        if os.path.isdir(args.checkpoint_path):
            checkpoint_model_type = load_model_type(args.checkpoint_path, default=args.model_type)
            if checkpoint_model_type != args.model_type:
                raise ValueError(
                    f"--model-type {args.model_type!r} does not match checkpoint model type "
                    f"{checkpoint_model_type!r} at {args.checkpoint_path}"
                )
            weights_path = os.path.join(args.checkpoint_path, "weights.pt")
        else:
            weights_path = args.checkpoint_path
        if args.pretrained is not None:
            print(f"NOTE: --checkpoint-path resume overwrites the --pretrained {args.pretrained} transfer"
                  " unless this checkpoint itself came from a --pretrained run.")
        model_state, optimizer_state = load_checkpoint(weights_path, map_location=device)
        model.load_state_dict(model_state)
        print(f"Loaded model from {weights_path}")
    model.train()

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.parameters() if not any(p is h for h in head_params)]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr, "weight_decay": args.head_weight_decay},
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale, "weight_decay": args.weight_decay},
    ])
    print(f"  Head LR: {args.lr:.1e} (wd {args.head_weight_decay})  "
          f"Backbone LR: {args.lr * args.backbone_lr_scale:.1e} (scale {args.backbone_lr_scale}, wd {args.weight_decay})")

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # load_state_dict restores the checkpoint's own param_group lrs (Adam moments
        # come along with them) — reassert this invocation's --lr/--backbone-lr-scale,
        # otherwise a resume silently ignores any new --lr and keeps training at the
        # old rate (OscillationAwareLR then captures those stale values as its base_lrs).
        for pg, lr in zip(optimizer.param_groups, (args.lr, args.lr * args.backbone_lr_scale)):
            pg["lr"] = lr
        print("  Loaded optimizer state (Adam moments) from checkpoint")
    elif args.checkpoint_path is not None:
        print("  WARNING: checkpoint has no saved optimizer state — Adam moments start cold, "
              "expect a transient loss spike over the first several post-resume steps")

    backbone_frozen = args.freeze_backbone_steps > 0
    if backbone_frozen:
        for p in backbone_params:
            p.requires_grad_(False)
        print(f"  Backbone frozen for first {args.freeze_backbone_steps} steps"
              f" ({sum(p.numel() for p in backbone_params)} params)")

    scheduler = OscillationAwareLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        factor=args.osc_factor,
        threshold=args.osc_threshold,
        cooldown=args.osc_cooldown,
        min_lr=args.osc_min_lr,
    )

    def save_weights(quiet=False):
        if any(p.isnan().any().item() for p in model.parameters()):
            print("WARNING: NaN in weights — skipping save")
            return
        os.makedirs(args.save_path, exist_ok=True)
        weights_path = os.path.join(args.save_path, "weights.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, weights_path)
        save_model_type(args.save_path, args.model_type)
        for src, dst_name in (
            ("src/constants.py", "constants.py"),
            (f"src/model/{args.model_type}/constants.py", "model_constants.py"),
        ):
            shutil.copy2(src, os.path.join(args.save_path, dst_name))
        if not quiet:
            print(f"Saved to {args.save_path}")

    def _sigint_handler(sig, frame):
        print("\nInterrupted — saving before exit...")
        save_weights()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    processor = SimpleImageProcessor(model_constants.img_size, augment=args.augment)
    data_args = types.SimpleNamespace(
        data_paths=args.data_path,
        data_weights=args.data_weights,
        image_folder=args.image_folder,
        image_processor=processor,
        image_aspect_ratio="pad",
    )
    dataset = SingleDataset(
        data_paths=data_args.data_paths,
        data_weights=data_args.data_weights,
        data_args=data_args,
    )

    if args.sample_size is not None:
        if args.sample_size > len(dataset):
            raise ValueError(
                f"--sample-size {args.sample_size} exceeds dataset size {len(dataset)}"
            )
        rng = random.Random(args.sample_seed)
        pool_indices = rng.sample(range(len(dataset)), args.sample_size)
        print(f"\nDataset: {args.sample_size} samples (subset of {len(dataset)}) from {args.data_path}"
              f" (weights {args.data_weights})")
    else:
        pool_indices = list(range(len(dataset)))
        print(f"\nDataset: {len(pool_indices)} samples from {args.data_path} (weights {args.data_weights})")

    if len(pool_indices) < args.batch_size:
        raise ValueError(f"Dataset too small: {len(pool_indices)} < batch_size {args.batch_size}")

    print(f"LR: {args.lr:.1e}  Batch: {args.batch_size}  Grad accum: {args.grad_accum}"
          f"  → effective batch: {args.batch_size * args.grad_accum}")
    print(f"Warmup steps: {args.warmup_steps}")

    print(f"\n{'Step':>5}  {'Loss':>10}  {'LR':>10}", flush=True)
    print("-" * 32, flush=True)

    plot_steps, plot_losses, plot_ema = [], [], []
    accum = args.grad_accum
    optimizer.zero_grad()
    running_loss = 0.0
    random.seed(args.sample_seed)
    train_started_at = perf_counter()

    for step in range(1, args.steps + 1):
        step_started_at = perf_counter()
        if backbone_frozen and step > args.freeze_backbone_steps:
            for p in backbone_params:
                p.requires_grad_(True)
            backbone_frozen = False
            scheduler.begin_backbone_warmup(step, group_idx=1, warmup_steps=args.backbone_warmup_steps)
            print(f"  [unfreeze] step {step}: backbone unfrozen (LR scale {args.backbone_lr_scale})")

        step_loss = 0.0
        for _ in range(accum):
            indices = random.sample(pool_indices, args.batch_size)
            batch = collate([dataset[i] for i in indices])
            images = batch["images"].to(device=device, dtype=torch.float32)
            level_probs = batch["level_probs"].to(device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), level_probs, reduction="batchmean") / accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(step_loss)

        completed_at = datetime.now().isoformat(timespec="seconds")
        step_elapsed = perf_counter() - step_started_at
        total_elapsed = perf_counter() - train_started_at
        print(f"  [perf] step {step}/{args.steps} completed_at {completed_at}"
              f"  step_sec {step_elapsed:.3f}  total_sec {total_elapsed:.3f}", flush=True)

        running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 1 else step_loss
        if step % args.log_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:>5}  {step_loss:>10.6f}  {current_lr:>10.2e}"
                  f"  (ema {running_loss:.4f})  [osc {scheduler.ema_osc:.3f}]", flush=True)
            plot_steps.append(step)
            plot_losses.append(step_loss)
            plot_ema.append(running_loss)
            if step > 1:
                save_weights(quiet=True)

    print("\nDone.", flush=True)
    save_weights()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.plot(plot_steps, plot_losses, alpha=0.4, color="steelblue", label="step loss")
    ax.plot(plot_steps, plot_ema, color="steelblue", linewidth=2, label="EMA loss")
    ax.legend()
    fig.tight_layout()
    os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.save_path))
    loss_plot_path = os.path.join(ANALYSIS_RESULTS_DIR, f"{run_name}_loss.png")
    fig.savefig(loss_plot_path, dpi=100)
    plt.close(fig)
    print(f"Loss curve saved to {loss_plot_path}", flush=True)
