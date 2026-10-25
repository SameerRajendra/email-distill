"""Aggregate results/scaling/<gpu>.jsonl into results/scaling/SCALING.md and scaling_efficiency.png.

Efficiency is tokens/s per GPU at N GPUs divided by the same strategy's tokens/s per GPU at 1 GPU (weak scaling:
the per-GPU batch is fixed, so 100% means adding GPUs cost nothing). Every number is computed from the JSONL.

    python report_scaling.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path("results/scaling")
STRATEGIES = {"ddp": ("#2a78d6", "o"), "fsdp": ("#eb6834", "s")}      # fixed colour + marker per strategy


def fmt(x, spec):
    return "n/a" if x is None else format(x, spec)


def main():
    runs = {p.stem: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            for p in sorted(DIR.glob("*.jsonl"))}
    if not runs:
        raise SystemExit("no results/scaling/*.jsonl: run `modal run train_dist_modal.py` first")
    first = next(r for rs in runs.values() for r in rs if "versions" in r)
    md = ["# Distributed LoRA fine-tuning: scaling", "",
          f"Gemma 4 E4B, LoRA r=16, bf16, real email rows, {first['per_gpu_batch']} rows per GPU per step "
          "(weak scaling). Throughput is real (non-padding) tokens per second over a fixed window of training steps; "
          "the comm share is NCCL kernel time over all CUDA kernel time on 2 profiled steps, rank 0. "
          "Versions: " + ", ".join(f"{k} {v}" for k, v in first["versions"].items()) + ".", ""]
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.8), squeeze=False, sharey=True)
    for ax, (name, rs) in zip(axes[0], runs.items()):
        gpu = next((r["gpu"] for r in rs if "gpu" in r), name)
        base = {r["strategy"]: r["tokens_per_s_per_gpu"] for r in rs if r["world_size"] == 1 and "error" not in r}
        md += [f"## {gpu}", "", "| Strategy | GPUs | Samples/s | Tokens/s per GPU | Efficiency | Peak GiB (max rank) | Comm % | Final loss |",
               "|---|---|---|---|---|---|---|---|"]
        curves = {s: [] for s in STRATEGIES}
        for r in sorted(rs, key=lambda r: (r["strategy"], r["world_size"])):
            if "error" in r:
                md.append(f"| {r['strategy']} | {r['world_size']} | failed: {r['error']} | | | | | |")
                continue
            eff = r["tokens_per_s_per_gpu"] / base[r["strategy"]] if r["strategy"] in base else None
            curves[r["strategy"]].append((r["world_size"], eff))
            md.append(f"| {r['strategy']} | {r['world_size']} | {r['samples_per_s']:.2f} | {r['tokens_per_s_per_gpu']:.0f} | "
                      f"{fmt(eff, '.1%')} | {fmt(r['peak_gib'], '.1f')} | {fmt(r['comm_share'], '.1%')} | {r['final_loss']:.3f} |")
        for s, pts in curves.items():
            pts = [(n, e) for n, e in pts if e is not None]
            if pts:
                ax.plot(*zip(*pts), color=STRATEGIES[s][0], marker=STRATEGIES[s][1], lw=2, ms=8, label=s.upper())
        ax.axhline(1, color="#8a8a85", lw=1, ls="--", label="ideal")
        ax.set(title=gpu, xlabel="GPUs", xticks=[1, 2, 4])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0][0].set_ylabel("scaling efficiency (tokens/s per GPU vs 1 GPU)")
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(DIR / "scaling_efficiency.png", dpi=150)
    md.append("![scaling efficiency](scaling_efficiency.png)")
    (DIR / "SCALING.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
