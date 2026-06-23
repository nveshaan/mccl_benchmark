"""
FSDP training benchmark on a synthetic task. Optimized for macOS / MPS using MCCL.

This script demonstrates FSDP (Fully Sharded Data Parallel) combined with MCCL
for high-performance multi-GPU or multi-Mac collective communications on Apple Silicon.

examples:
    # 2-rank FSDP on one Mac with MCCL
    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
        src/fsdp_train.py --save-stats bench_runs/

    # 2-node FSDP setup over a physical Thunderbolt bridge
    # Run on Mac 0 (Master Node):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=192.168.1.50 --master_port=29500 \
        src/fsdp_train.py --save-stats bench_runs/

    # Run on Mac 1 (Worker Node):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
        --master_addr=192.168.1.50 --master_port=29500 \
        src/fsdp_train.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# Import from ddp_utils
from utils import (
    SyntheticDataset,
    build_model,
    run_config_from_args,
    write_training_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP training benchmark with MCCL")
    p.add_argument("--steps", type=int, default=100, help="Training steps after warmup")
    p.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size")
    p.add_argument("--input-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--stress-model", action="store_true", help="Use large model defaults")
    p.add_argument("--fp16", action="store_true", help="Enable Mixed Precision fp16 for FSDP")
    p.add_argument("--save-stats", metavar="PATH", default=None, help="Write timing JSON")
    p.add_argument("--warmup", type=int, default=5, help="Warmup steps before timing")
    return p.parse_args()


def get_model_dims(args) -> tuple[int, int, int, int]:
    if args.stress_model:
        return (
            args.input_dim if args.input_dim != 512 else 2048,
            args.num_classes if args.num_classes != 64 else 128,
            args.hidden if args.hidden != 1024 else 8192,
            args.depth if args.depth != 4 else 16,
        )
    return args.input_dim, args.num_classes, args.hidden, args.depth


def run_fsdp(args) -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.backends.mps.is_available():
        print("MPS not available (required for Mac-based benchmarks).", file=sys.stderr)
        sys.exit(1)

    # Load and initialize MCCL optimized defaults
    import mccl
    mccl.apply_thunderbolt_production_defaults(training_defaults=True)
    _setup_mccl_env()

    # Assign MPS device and initialize distributed process group with MCCL
    device = torch.device("mps:0")
    dist.init_process_group(backend="mccl", device_id=device)

    if rank == 0:
        print(f"[fsdp] rank={rank} world_size={world_size} device={device}", flush=True)

    torch.manual_seed(42)
    dims = get_model_dims(args)
    model = build_model(*dims).to(device)
    torch.manual_seed(42 + rank)

    # Compute total parameters BEFORE wrapping with FSDP
    total_params = sum(p.numel() for p in model.parameters())

    # Configure Mixed Precision policy if fp16 is enabled
    mp_policy = None
    if args.fp16:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # Wrap model with FSDP
    # Sharding Strategy: FULL_SHARD shards parameters + gradients + optimizer states across ranks
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    dataset = SyntheticDataset(dims[0], dims[1])

    if rank == 0:
        print(f"FSDP training | backend=mccl world_size={world_size}\n"
              f"  Model: {total_params:,} parameters\n"
              f"  Batch: {args.batch_size}/rank ({args.batch_size * world_size} global)\n"
              f"  Steps: {args.steps}"
              f"{'  fp16' if args.fp16 else ''}", flush=True)

    step_times: list[float] = []
    losses: list[float] = []

    for step in range(args.warmup + args.steps):
        x, y = dataset.get_batch(args.batch_size, device, step, rank, world_size)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(fsdp_model(x), y)
        loss.backward()
        optimizer.step()
        
        torch.mps.synchronize()
        dt = time.perf_counter() - t0

        if step >= args.warmup:
            step_times.append(dt)
            losses.append(loss.item())
        if rank == 0 and (step % 5 == 0 or step == args.warmup + args.steps - 1):
            tag = "warmup" if step < args.warmup else "train"
            print(f"  {tag} {step:4d}  loss={loss.item():.6f}  {dt:.3f}s", flush=True)

    # Print Final Performance Results and capture hardware-level MCCL metrics
    if step_times and rank == 0:
        avg = sum(step_times) / len(step_times)
        gbs = args.batch_size * world_size

        mccl_info = ""
        try:
            import mccl as _m
            metrics = _m.get_metrics()
            if metrics:
                mccl_info = (f"\n  MCCL: {metrics.total_ops} ops, "
                             f"avg_lat={metrics.avg_latency_ms:.2f}ms")
                for attr in ("avg_sync_ms", "avg_network_ms", "avg_reduce_ms"):
                    val = getattr(metrics, attr, None)
                    if val is not None:
                        mccl_info += f"  {attr}={val:.2f}ms"
        except Exception:
            pass

        print(f"\n=== FSDP Stats ===\n"
              f"  Avg: {avg:.3f}s ({1/avg:.1f} steps/s)\n"
              f"  Min/Max: {min(step_times):.3f}s / {max(step_times):.3f}s\n"
              f"  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}\n"
              f"  Throughput: {gbs/avg:.0f} samples/s{mccl_info}", flush=True)

    # Save metrics using the format we updated for plot.py compatibility
    if args.save_stats and rank == 0 and step_times:
        cfg = run_config_from_args(args, baseline=False)
        cfg["backend"] = "mccl"
        
        mccl_metrics = None
        try:
            import mccl as _m
            metrics = _m.get_metrics()
            if metrics:
                mccl_metrics = {
                    "total_ops": metrics.total_ops,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "avg_sync_ms": getattr(metrics, "avg_sync_ms", None),
                    "avg_network_ms": getattr(metrics, "avg_network_ms", None),
                    "avg_reduce_ms": getattr(metrics, "avg_reduce_ms", None),
                }
        except Exception:
            pass

        out = write_training_stats(
            args.save_stats, f"fsdp_mccl_ws{world_size}", step_times, losses,
            args.batch_size, world_size, total_params,
            config=cfg, mccl_metrics=mccl_metrics,
        )
        print(f"Wrote stats to {out}", flush=True)

    dist.destroy_process_group()


def _setup_mccl_env() -> None:
    """Configures system ports and loopback addresses for MCCL runtime context."""
    if "MCCL_PORT_BASE" not in os.environ:
        mp = int(os.environ.get("MASTER_PORT", "29500"))
        os.environ["MCCL_PORT_BASE"] = str(mp + 100)
    master = os.environ.get("MASTER_ADDR", "")
    if master in ("127.0.0.1", "localhost", "::1") and "MCCL_LISTEN_ADDR" not in os.environ:
        os.environ["MCCL_LISTEN_ADDR"] = "127.0.0.1"


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        run_fsdp(args)
    else:
        print("Launch with torchrun for FSDP:\n"
              "  torchrun --nproc_per_node=2 src/fsdp_train.py", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()