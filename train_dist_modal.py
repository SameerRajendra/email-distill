"""Distributed LoRA fine-tuning of Gemma 4 E4B on Modal: a DDP-vs-FSDP scaling grid, then the final run.

Multi-GPU fine-tuning cost depends on the parallelism strategy and on the interconnect (PCIe L40S vs NVLink H100).
The grid runs ddp and fsdp on 1, 2 and 4 GPUs of each type (one container per type and count, all started at once),
writes results/scaling/<gpu>.jsonl, and `python report_scaling.py` turns those into tables and a plot.
`--final` trains the 2-epoch model on 4 GPUs with the best measured strategy, then merges it and exports the same
GGUFs as train_modal.py into the Volume, so eval_modal.py works unchanged.

    modal run train_dist_modal.py --gpus l40s --counts 1,2      # cheap smoke test first
    modal run train_dist_modal.py                               # the grid: l40s,h100 x 1,2,4
    modal run train_dist_modal.py --final --final-gpu h100      # add --strategy ddp|fsdp to override the pick

Privacy: the rows travel as a function argument, as in train_modal.py. torchrun's workers need them in a file, which
is written to the container's own /tmp (never a Volume) and disappears with the container.
"""
import json
import shutil
import subprocess
from pathlib import Path

import modal

MODEL = "unsloth/gemma-4-E4B-it"
VOLUME, HF_VOLUME = "email-distill-models", "email-distill-hf-cache"
LLAMA_CPP = "/opt/llama.cpp"
OUT = "/tmp/out.jsonl"
STRATEGIES = ["ddp", "fsdp"]
GPU = {"l40s": "L40S", "h100": "H100!"}     # "!" keeps Modal from silently upgrading an H100 request to an H200
PRICE = {"l40s": 0.000542, "h100": 0.001097}    # USD per GPU-second, Modal list prices (modal.com/pricing, 2026-10-24)
CPU, GIB = 8, 64                                # per container; billed at 0.0000131 per core-second, 0.00000222 per GiB-second
HOST = CPU * 0.0000131 + GIB * 0.00000222
SCALING_MIN, FINAL_MIN, EXPORT_MIN = 15, 20, 45     # function timeouts, which cap the spend
ASSUMED_MIN, OVERHEAD_MIN = 6, 5    # guesses, not measurements: minutes per scaling container; startup, eval and save of the final

base = (modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install("transformers==5.20.2", "trl==1.16.0", "peft==0.21.2", "accelerate==1.15.0", "datasets==5.1.0")
        .env({"HF_HOME": "/hf/cache"}))
train_image = base.env({"HF_HUB_OFFLINE": "1"}).add_local_file(Path(__file__).parent / "train_dist.py", "/app/train_dist.py")
# llama.cpp is built CPU-only (quantizer only), as in train_modal.py.
export_image = (base.apt_install("git", "build-essential", "cmake").pip_install("sentencepiece", "protobuf")
                .run_commands(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP}",
                              f"cmake -S {LLAMA_CPP} -B {LLAMA_CPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
                              f"cmake --build {LLAMA_CPP}/build --target llama-quantize -j"))
app = modal.App("email-distill-dist")
hf_cache = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)
models = modal.Volume.from_name(VOLUME, create_if_missing=True)
container = dict(cpu=CPU, memory=GIB * 1024)


def torchrun(strategy: str, rows: dict, *extra: str) -> int:
    """Run train_dist.py on every GPU of this container and return the exit code."""
    import torch
    Path("/tmp/rows.json").write_text(json.dumps(rows), encoding="utf-8")
    return subprocess.run(["torchrun", "--standalone", f"--nproc_per_node={torch.cuda.device_count()}", "/app/train_dist.py",
                           "--strategy", strategy, "--data", "/tmp/rows.json", "--out", OUT, "--model", MODEL, *extra]).returncode


@app.function(image=base, volumes={"/hf": hf_cache}, timeout=30 * 60)
def prefetch():
    """Download the weights once, before any GPU container starts, so the parallel jobs only read the Volume."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)
    hf_cache.commit()


@app.function(image=train_image, volumes={"/hf": hf_cache}, timeout=SCALING_MIN * 60, **container)
def scaling(rows: dict) -> list[str]:
    """Every strategy on this container's GPUs (the count comes from .with_options(gpu=...)). Returns JSONL lines."""
    import torch
    for strategy in STRATEGIES:
        code = torchrun(strategy, rows)
        if code:    # an OOM, say, is a result too: record it and carry on with the next strategy
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"strategy": strategy, "world_size": torch.cuda.device_count(), "error": f"torchrun exit {code}"}) + "\n")
    return Path(OUT).read_text(encoding="utf-8").splitlines()


@app.function(image=train_image, volumes={"/hf": hf_cache, "/out": models}, timeout=FINAL_MIN * 60, **container)
def train_final(rows: dict, strategy: str) -> list[dict]:
    """2 epochs on this container's GPUs; only the LoRA adapter is written to the Volume. Returns the loss log."""
    shutil.rmtree("/out/adapter", ignore_errors=True)       # drop the adapter of any earlier run
    if torchrun(strategy, rows, "--final", "/out/adapter"):
        raise RuntimeError("final training failed, see the log")
    models.commit()
    return json.loads(Path(OUT).read_text(encoding="utf-8").splitlines()[-1])["log_history"]


@app.function(image=export_image, volumes={"/hf": hf_cache, "/out": models}, timeout=EXPORT_MIN * 60, **container)
def export():
    """Merge the adapter on CPU (no GPU billed), convert to BF16 GGUF and quantize, as train_modal.py does."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    PeftModel.from_pretrained(base_model, "/out/adapter").merge_and_unload().save_pretrained("/tmp/merged")
    AutoTokenizer.from_pretrained(MODEL).save_pretrained("/tmp/merged")
    subprocess.run(["python", f"{LLAMA_CPP}/convert_hf_to_gguf.py", "/tmp/merged",
                    "--outfile", "/tmp/bf16.gguf", "--outtype", "bf16"], check=True)
    for quant in ("Q4_K_M", "Q8_0"):
        subprocess.run([f"{LLAMA_CPP}/build/bin/llama-quantize", "/tmp/bf16.gguf", f"/out/email-distill-{quant}.gguf", quant],
                       check=True)
    models.commit()


def cost(gpu: str, n: int, minutes: float) -> float:
    return minutes * 60 * (n * PRICE[gpu] + HOST)


def best(gpu: str):
    """(strategy, samples/s) of the fastest 4-GPU run in results/scaling/<gpu>.jsonl, or None."""
    path = Path("results/scaling") / f"{gpu}.jsonl"
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return max(((r["strategy"], r["samples_per_s"]) for r in runs if r["world_size"] == 4 and "samples_per_s" in r),
               key=lambda x: x[1], default=None)


@app.local_entrypoint()
def main(gpus: str = "l40s,h100", counts: str = "1,2,4", final: bool = False, final_gpu: str = "h100", strategy: str = ""):
    def rows(name):
        return [json.loads(line) for line in (Path(__file__).parent / "data" / name).read_text(encoding="utf-8").splitlines()]

    train = rows("train.jsonl")
    if not final:
        jobs = [(g, int(n)) for g in gpus.split(",") for n in counts.split(",")]
        gpu_min = sum(n for _, n in jobs)
        print(f"Scaling grid: {len(jobs)} containers, {gpu_min} GPUs in all, list prices {PRICE} USD per GPU-second.\n"
              f"  GPU-minutes: {gpu_min * ASSUMED_MIN} at an assumed {ASSUMED_MIN} min per container "
              f"-> ${sum(cost(g, n, ASSUMED_MIN) for g, n in jobs):.2f} (a guess); at most {gpu_min * SCALING_MIN} at the "
              f"{SCALING_MIN} min timeout -> ${sum(cost(g, n, SCALING_MIN) for g, n in jobs):.2f} (the cap).")
        prefetch.remote()
        calls = [(g, n, scaling.with_options(gpu=f"{GPU[g]}:{n}").spawn({"train": train})) for g, n in jobs]
        out = Path("results/scaling")
        out.mkdir(parents=True, exist_ok=True)
        for g in gpus.split(","):
            (out / f"{g}.jsonl").write_text("")
        for g, n, call in calls:
            try:
                lines = call.get()
            except Exception as e:      # keep the other jobs' results
                print(f"{g} x{n} failed: {e}")
                continue
            with (out / f"{g}.jsonl").open("a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
            print(f"{g} x{n} done")
        return

    print(f"Final run, 2 epochs on 4 GPUs ({2 * len(train)} samples), then CPU export (cap ${EXPORT_MIN * 60 * HOST:.2f}):")
    for g in PRICE:
        pick = best(g)
        guess = f"{pick[0]} at {pick[1]:.2f} samples/s measured -> " \
                f"${cost(g, 4, 2 * len(train) / pick[1] / 60 + OVERHEAD_MIN):.2f} incl. {OVERHEAD_MIN} min assumed overhead" \
                if pick else "no measured 4-GPU result yet, so no estimate"
        print(f"  {g}: {guess}; cap ${cost(g, 4, FINAL_MIN):.2f} at the {FINAL_MIN} min timeout")
    pick = best(final_gpu)
    if not (strategy or pick):
        raise SystemExit("Pass --strategy ddp|fsdp, or run the scaling grid first so the fastest one is picked.")
    strategy = strategy or pick[0]
    print(f"Training with {strategy} on {final_gpu} x4.")
    prefetch.remote()
    for h in train_final.with_options(gpu=f"{GPU[final_gpu]}:4").remote({"train": train, "test": rows("test.jsonl")}, strategy):
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items() if k in ("epoch", "step", "loss", "eval_loss")})
    export.remote()
    print("\nFetch the results (run in this folder):")
    print(f"  modal volume get {VOLUME} email-distill-Q4_K_M.gguf models\\")
    print(f"  modal volume get {VOLUME} email-distill-Q8_0.gguf models\\")
    print(f"  modal volume get {VOLUME} adapter models\\adapter")
