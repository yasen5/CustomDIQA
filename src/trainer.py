import os
import random
import signal
import sys
import types

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.datasets.single_dataset import SingleDataset
from src.model import constants as model_constants
from src.model.model import EncoderModel


class SimpleImageProcessor:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    crop_size = {"height": model_constants.img_size, "width": model_constants.img_size}

    def __init__(self):
        self._transform = transforms.Compose([
            transforms.Resize((model_constants.img_size, model_constants.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std),
        ])

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

    @property
    def ema_osc(self):
        return self._ema_osc

    def step(self, loss):
        self._global_step += 1
        if self._global_step <= self.warmup_steps:
            scale = self._global_step / max(1, self.warmup_steps)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
            return
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

    model = EncoderModel().to(device=device, dtype=torch.float32)
    model.train()

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.parameters() if not any(p is h for h in head_params)]
    if args.backbone_lr_scale != 1.0:
        optimizer = torch.optim.Adam([
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        ])
        print(f"  Head LR: {args.lr:.1e}  Backbone LR: {args.lr * args.backbone_lr_scale:.1e}"
              f" (scale {args.backbone_lr_scale})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        torch.save(model.state_dict(), args.save_path)
        if not quiet:
            print(f"Saved to {args.save_path}")

    def _sigint_handler(sig, frame):
        print("\nInterrupted — saving before exit...")
        save_weights()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    processor = SimpleImageProcessor()
    data_args = types.SimpleNamespace(
        data_paths=[args.data_path],
        data_weights=[1],
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
        print(f"\nDataset: {args.sample_size} samples (subset of {len(dataset)}) from {args.data_path}")
    else:
        pool_indices = list(range(len(dataset)))
        print(f"\nDataset: {len(pool_indices)} samples from {args.data_path}")

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

    for step in range(1, args.steps + 1):
        step_loss = 0.0
        for _ in range(accum):
            indices = random.sample(pool_indices, args.batch_size)
            batch = collate([dataset[i] for i in indices])
            images = batch["images"].to(device=device, dtype=torch.float32)
            level_probs = batch["level_probs"].to(device=device)
            logits = model(images)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), level_probs, reduction="batchmean") / accum
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(step_loss)

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
    loss_plot_path = os.path.splitext(args.save_path)[0] + "_loss.png"
    fig.savefig(loss_plot_path, dpi=100)
    plt.close(fig)
    print(f"Loss curve saved to {loss_plot_path}", flush=True)
