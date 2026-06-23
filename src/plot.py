# Source: https://github.com/mps-ddp/mccl/blob/master/examples/benchmark_throughput.py
"""
Compare training throughput from ``--save-stats`` JSON files produced by
``src/ddp_train.py``.

Save runs into a directory (auto-named from CLI settings)::

    mkdir -p bench_runs

    python src/ddp_train.py --baseline --save-stats bench_runs/

    torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \\
        src/ddp_train.py --save-stats bench_runs/

    torchrun ... src/ddp_train.py --bucket-mb 100 --fp16 --save-stats bench_runs/

Plot every ``*.json`` in that directory::

    python src/plot.py bench_runs/ -o bench

Legacy two-file mode still works::

    python src/plot.py \\
        --baseline baseline_stats.json --ddp ddp_stats.json -o throughput_bench

Writes ``<output>.npz`` (NumPy arrays) and ``<output>.png`` / ``<output>_bars.png``
if matplotlib is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from utils import run_label_from_payload

_REQUIRED_KEYS = (
    "step_times",
    "losses",
    "avg_step_time_s",
    "throughput_samples_per_sec",
    "batch_size_per_rank",
    "world_size",
    "global_batch_size",
    "total_params",
    "mode",
)


@dataclass(frozen=True)
class RunStats:
    path: Path
    name: str
    label: str
    data: dict

    @property
    def throughput(self) -> float:
        return float(self.data["throughput_samples_per_sec"])

    @property
    def is_baseline(self) -> bool:
        return self.data["mode"] == "baseline"

    @property
    def global_batch(self) -> int:
        return int(self.data["global_batch_size"])

    @property
    def total_params(self) -> int:
        return int(self.data["total_params"])


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _validate(data: dict, source: Path) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")


def _load_runs(paths: list[Path]) -> list[RunStats]:
    runs: list[RunStats] = []
    for path in sorted(paths):
        data = _load(path)
        _validate(data, path)
        
        # Get the base descriptive label from ddp_utils
        base_label = run_label_from_payload(data)
        
        # Safe extraction of the world_size integer from the JSON payload
        ws = data.get("world_size", 1)
        
        # Append worldsize cleanly depending on count
        rank_suffix = f" ({ws} rank)" if ws == 1 else f" ({ws} ranks)"
        final_label = f"{base_label}{rank_suffix}"
        
        runs.append(
            RunStats(
                path=path,
                name=path.stem,
                label=final_label,  # Injects the new string configuration
                data=data,
            )
        )
    return runs


def _discover_jsons(stats_dir: Path) -> list[Path]:
    if not stats_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {stats_dir}")
    files = sorted(stats_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no *.json files in {stats_dir}")
    return files


def _print_table(runs: list[RunStats], baselines: list[RunStats]) -> None:
    baseline_tput = baselines[0].throughput if baselines else None
    print("\n=== Runs ===")
    print(f"{'file':<40} {'mode':<12} {'throughput':>12}  {'global_batch':>12}  notes")
    print("-" * 95)
    for run in runs:
        ratio = ""
        if baseline_tput and not run.is_baseline and baseline_tput > 0:
            pct = run.throughput / baseline_tput * 100.0
            ratio = f"  ({pct:.0f}% of baseline)"
        cfg = run.data.get("config") or {}
        notes = []
        if cfg.get("bucket_mb") is not None:
            notes.append(f"bucket={cfg['bucket_mb']}MB")
        if cfg.get("fp16"):
            notes.append("fp16")
        if cfg.get("stress_model"):
            notes.append("stress")
        if cfg.get("backend") and cfg["backend"] != "mccl":
            notes.append(cfg["backend"])
        note_str = ", ".join(notes) + ratio
        print(
            f"{run.name:<40} {run.data['mode']:<12} "
            f"{run.throughput:>10,.1f}  {run.global_batch:>12}  {note_str}"
        )


def _pick_baseline(runs: list[RunStats]) -> list[RunStats]:
    baselines = [r for r in runs if r.is_baseline]
    if baselines:
        return baselines
    return []


def _matching_baseline(run: RunStats, baselines: list[RunStats]) -> RunStats | None:
    for bl in baselines:
        if (
            bl.global_batch == run.global_batch
            and bl.total_params == run.total_params
        ):
            return bl
    return baselines[0] if baselines else None


def _save_npz(runs: list[RunStats], out_npz: Path) -> None:
    import numpy as np

    payload: dict = {"run_names": np.array([r.name for r in runs], dtype=object)}
    for run in runs:
        prefix = run.name.replace("-", "_")
        payload[f"{prefix}__step_times"] = np.asarray(run.data["step_times"], dtype=np.float64)
        payload[f"{prefix}__losses"] = np.asarray(run.data["losses"], dtype=np.float64)
        payload[f"{prefix}__throughput"] = run.throughput
        payload[f"{prefix}__global_batch"] = run.global_batch
    np.savez(out_npz, **payload)


def _plot(runs: list[RunStats], baselines: list[RunStats], output: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "(matplotlib not installed; skipped .png — pip install matplotlib)",
            file=sys.stderr,
        )
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    baseline_tput = baselines[0].throughput if baselines else None

    # ── Step times + loss (all runs) ─────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for idx, run in enumerate(runs):
        times = np.asarray(run.data["step_times"], dtype=np.float64)
        losses = np.asarray(run.data["losses"], dtype=np.float64)
        mean_ms = float(np.mean(times) * 1000.0)
        axes[0].plot(
            np.arange(len(times)),
            times * 1000.0,
            label=f"{run.label} ({mean_ms:.0f} ms)",
            alpha=0.85,
            linewidth=1.0,
            color=colors[idx % len(colors)],
        )
        axes[1].plot(
            np.cumsum(times),
            losses,
            label=run.label,
            alpha=0.85,
            linewidth=1.0,
            color=colors[idx % len(colors)],
        )

    axes[0].set_ylabel("Step time (ms)")
    axes[0].set_title("Per-step wall time")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 100)

    axes[1].set_xlabel("Wall time (s, cumulative from timed train steps)")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 500)

    nparams = runs[0].total_params
    gbatch = runs[0].global_batch
    title_extra = ""
    if baseline_tput and len(runs) >= 2:
        best_ddp = max((r.throughput for r in runs if not r.is_baseline), default=0.0)
        if best_ddp > baseline_tput:
            title_extra = f"  |  best DDP = {best_ddp / baseline_tput:.2f}× baseline"
        elif best_ddp > 0:
            title_extra = f"  |  baseline / best DDP = {baseline_tput / best_ddp:.2f}×"
    fig.suptitle(
        f"MCCL benchmark ({len(runs)} runs)  |  global_batch={gbatch}  |  "
        f"~{nparams / 1e6:.1f}M params{title_extra}",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_png = Path(f"{output}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png.resolve()}")

    # ── Throughput bar chart ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(8, len(runs) * 0.9), 5))
    labels = [r.label.replace(" ", "\n") for r in runs]
    vals = [r.throughput for r in runs]
    bar_colors = [
        "#2ecc71" if r.is_baseline else "#3498db" for r in runs
    ]
    bars = ax2.bar(range(len(runs)), vals, color=bar_colors, width=0.65)
    ax2.set_xticks(range(len(runs)))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Throughput (samples / sec)")
    ax2.set_title(f"Average throughput  |  {len(runs)} runs")
    ax2.grid(True, axis="y", alpha=0.3)
    ymax = max(vals) * 1.22 if vals else 1.0
    ax2.set_ylim(0, ymax)
    for bar, run, v in zip(bars, runs, vals, strict=True):
        note = f"{v:.1f}"
        bl = _matching_baseline(run, baselines)
        if bl and not run.is_baseline and bl.throughput > 0:
            note += f"\n({v / bl.throughput * 100:.0f}%)"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            note,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig2.tight_layout()
    out_bar = Path(f"{output}_bars.png")
    fig2.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out_bar.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ddp_train.py --save-stats JSON files.",
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        type=Path,
        help="Directory containing *.json stats files",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="(legacy) Single baseline JSON file",
    )
    parser.add_argument(
        "--ddp",
        type=Path,
        help="(legacy) Single DDP JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plot",
        metavar="PREFIX",
        help="Output prefix for .npz and .png (default: plot)",
    )
    args = parser.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("error: numpy is required (pip install numpy)", file=sys.stderr)
        sys.exit(1)

    if args.stats_dir is not None:
        paths = _discover_jsons(args.stats_dir)
    elif args.baseline and args.ddp:
        for label, p in ("baseline", args.baseline), ("ddp", args.ddp):
            if not p.is_file():
                print(f"error: {label} file not found: {p}", file=sys.stderr)
                sys.exit(1)
        paths = [args.baseline, args.ddp]
    else:
        parser.error("provide STATS_DIR or both --baseline and --ddp")

    try:
        runs = _load_runs(paths)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    baselines = _pick_baseline(runs)
    _print_table(runs, baselines)
    print("\n")
    _plot(runs, baselines, args.output)


if __name__ == "__main__":
    main()
