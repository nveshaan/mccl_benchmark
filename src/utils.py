# Source: https://github.com/mps-ddp/mccl/blob/master/examples/ddp_utils.py
"""Shared utilities for DDP training examples: models, data, stats."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


# ── Synthetic dataset ────────────────────────────────────────────────

class SyntheticDataset:
    """Well-separated Gaussian clusters -- trivially learnable for any model.

    Fixed class centroids spread far apart. Each batch picks a random class
    per sample, adds small Gaussian noise, and returns (x, y). Identical
    centroids on every rank; per-step RNG gives different samples per rank.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 seed: int = 424242, separation: float = 5.0) -> None:
        g = torch.Generator()
        g.manual_seed(seed)
        self.centroids = torch.randn(num_classes, input_dim, generator=g) * separation
        self.input_dim = input_dim
        self.num_classes = num_classes

    def get_batch(self, batch_size: int, device: torch.device,
                  step: int, rank: int, world_size: int,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(1000 + step * world_size + rank)
        y = torch.randint(0, self.num_classes, (batch_size,))
        centroids = self.centroids.to(device=device)
        noise = torch.randn(batch_size, self.input_dim, device=device) * 0.3
        x = centroids[y] + noise
        y = y.to(device=device)
        return x, y


# ── Model components ─────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.w_o(out)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, large_kernels: bool = True):
        super().__init__()
        k = (7, 5, 9, 11) if large_kernels else (3, 3, 3, 3)
        p = (3, 2, 4, 5) if large_kernels else (1, 1, 1, 1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k[0], padding=p[0])
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k[1], padding=p[1])
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=k[2], padding=p[2])
        self.depthwise = nn.Conv2d(out_ch, out_ch, kernel_size=k[3], padding=p[3], groups=out_ch)
        self.pointwise = nn.Conv2d(out_ch, out_ch, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.bn5 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)

        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(out_ch, out_ch // 4)
        self.se_fc2 = nn.Linear(out_ch // 4, out_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.dropout(self.gelu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.depthwise(x)))
        x = self.gelu(self.bn5(self.pointwise(x)))
        b, c, _, _ = x.shape
        se = self.sigmoid(self.se_fc2(self.relu(self.se_fc1(self.se_pool(x).view(b, c)))))
        x = x * se.view(b, c, 1, 1)
        if identity.shape == x.shape:
            x = x + identity
        return x


class HeavyDummyModel(nn.Module):
    """Conv + attention + MLP model (~96M params at default dims)."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int, depth: int):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(input_dim, hidden)
        self.conv_proj = nn.Linear(hidden, 64 * 64 * 16)

        self.conv1 = ConvBlock(16, 32, large_kernels=True)
        self.conv2 = ConvBlock(32, 64, large_kernels=True)
        self.conv3 = ConvBlock(64, 64, large_kernels=True)

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=8, dilation=8), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        self.attn_proj = nn.Linear(64 * 16 * 16, hidden)
        self.seq_len = 16
        self.pos_encoding = nn.Parameter(torch.randn(self.seq_len, hidden))
        self.attn1 = MultiHeadAttention(hidden, n_heads=8)
        self.attn2 = MultiHeadAttention(hidden, n_heads=8)

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.input_proj(x)
        cx = self.conv_proj(x).view(B, 16, 64, 64)
        cx = self.dilated_conv(self.conv3(self.conv2(self.conv1(cx))))
        cx = self.pool(cx).flatten(1)
        ax = self.attn_proj(cx).unsqueeze(1).expand(B, self.seq_len, self.hidden)
        ax = ax + self.pos_encoding.unsqueeze(0)
        ax = self.attn1(ax) + ax
        ax = self.attn2(ax) + ax
        return self.mlp(ax.mean(dim=1))


def build_model(input_dim: int, num_classes: int, hidden: int, depth: int) -> nn.Module:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return HeavyDummyModel(input_dim, num_classes, hidden, depth)


# ── Stats I/O ────────────────────────────────────────────────────────

def run_config_from_args(args, *, baseline: bool) -> dict:
    """Snapshot of ``ddp_train.py`` CLI settings for JSON metadata."""
    dims = (
        (
            args.input_dim if args.input_dim != 512 else 2048,
            args.num_classes if args.num_classes != 64 else 128,
            args.hidden if args.hidden != 1024 else 8192,
            args.depth if args.depth != 4 else 16,
        )
        if args.stress_model
        else (args.input_dim, args.num_classes, args.hidden, args.depth)
    )
    return {
        "baseline": baseline,
        "backend": None if baseline else args.backend,
        "steps": args.steps,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "bucket_mb": None if baseline else args.bucket_mb,
        "fp16": args.fp16,
        "stress_model": args.stress_model,
        "input_dim": dims[0],
        "num_classes": dims[1],
        "hidden": dims[2],
        "depth": dims[3],
    }


def build_stats_slug(mode: str, config: dict) -> str:
    """Filesystem-safe name describing a run (no ``.json`` suffix)."""
    parts = [mode.replace("_", "-")]
    if config.get("backend") and config["backend"] != "mccl":
        parts.append(config["backend"])
    parts.append(f"batch{config['batch_size']}")
    parts.append(f"steps{config['steps']}")
    if config.get("bucket_mb") is not None:
        parts.append(f"bucket{config['bucket_mb']}")
    if config.get("fp16"):
        parts.append("fp16")
    if config.get("stress_model"):
        parts.append("stress")
    for key, default in (("hidden", 1024), ("depth", 4), ("input_dim", 512)):
        val = config.get(key)
        if val is not None and val != default and not config.get("stress_model"):
            parts.append(f"{key[:1]}{val}")
    return "__".join(parts)


def resolve_stats_path(save_stats: str | os.PathLike[str], slug: str) -> Path:
    """Return a file path; create parent dir when ``save_stats`` is a directory."""
    dest = Path(save_stats)
    if dest.suffix == ".json":
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    return dest / f"{slug}.json"


def run_label_from_payload(payload: dict) -> str:
    """Short human-readable label for charts and tables."""
    mode = payload.get("mode", "run")
    cfg = payload.get("config") or {}
    parts = [mode.replace("_", " ")]
    if cfg.get("backend") and cfg["backend"] != "mccl":
        parts.append(cfg["backend"])
    if cfg.get("bucket_mb") is not None:
        parts.append(f"bucket={cfg['bucket_mb']}MB")
    if cfg.get("fp16"):
        parts.append("fp16")
    if cfg.get("stress_model"):
        parts.append("stress")
    return " ".join(parts)


def write_training_stats(
    path: str | os.PathLike[str],
    mode: str,
    step_times: list[float],
    losses: list[float],
    batch_size: int,
    world_size: int,
    total_params: int,
    *,
    config: dict | None = None,
    mccl_metrics: dict | None = None,
) -> Path:
    avg_time = sum(step_times) / len(step_times) if step_times else 0.0
    global_batch = batch_size * world_size
    throughput = global_batch / avg_time if avg_time > 0 else 0.0

    slug = build_stats_slug(mode, config) if config else mode
    out_path = resolve_stats_path(path, slug)

    payload: dict = {
        "mode": mode,
        "step_times": step_times,
        "losses": losses,
        "avg_step_time_s": avg_time,
        "throughput_samples_per_sec": throughput,
        "batch_size_per_rank": batch_size,
        "world_size": world_size,
        "global_batch_size": global_batch,
        "total_params": total_params,
    }
    if config is not None:
        payload["config"] = config
    if mccl_metrics is not None:
        payload["mccl_metrics"] = mccl_metrics

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
