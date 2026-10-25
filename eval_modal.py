"""Run eval.py's measurement on a Modal L4 GPU, so the GGUFs never have to be downloaded to the PC.

The tuned GGUFs are read from the Volume `email-distill-models` (written by train_modal.py); the stock model
is downloaded once from Hugging Face into a second Volume. The test rows travel as a function argument
(memory only), never to a Volume, like the training rows in train_modal.py. Writes results/<tag>.json here.

    modal run eval_modal.py
    modal run eval_modal.py --tags tuned-q4_k_m
"""
import json
import subprocess
from pathlib import Path

import modal

HF_DIR = "/hf"
SLOTS = 4   # parallel llama-server slots for the JSON-validity pass (L4 has room; the laptop run uses 1)
MODELS = {  # tag -> GGUF path inside the container
    "base-q4_k_m": f"{HF_DIR}/gemma-4-E4B-it-Q4_K_M.gguf",
    "tuned-q4_k_m": "/models/email-distill-Q4_K_M.gguf",
    "tuned-q8_0": "/models/email-distill-Q8_0.gguf",
}

# Official prebuilt llama.cpp (CUDA 12) server image: llama-server is /app/llama-server. Its ENTRYPOINT is
# that binary, which would swallow Modal's own start command, so it is cleared.
image = (modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
         .entrypoint([])
         .pip_install("rapidfuzz", "huggingface_hub")
         .add_local_python_source("eval", "schema"))
app = modal.App("email-distill-eval", image=image)
models = modal.Volume.from_name("email-distill-models")
hf_cache = modal.Volume.from_name("email-distill-hf-cache", create_if_missing=True)


@app.function(gpu="L4", cpu=4, timeout=90 * 60, volumes={"/models": models.read_only(), HF_DIR: hf_cache})
def evaluate_tag(tag: str, rows: list[dict]) -> dict:
    from huggingface_hub import hf_hub_download

    from eval import evaluate, serve

    model = MODELS[tag]
    if model.startswith(HF_DIR):
        hf_hub_download("lmstudio-community/gemma-4-E4B-it-GGUF", Path(model).name, local_dir=HF_DIR)
        hf_cache.commit()
    log = Path(f"/tmp/{tag}.server.log")
    try:
        with serve("/app/llama-server", model, log, slots=SLOTS) as url:
            res = evaluate(url, tag, model, rows, slots=SLOTS)
    finally:   # the server log only exists in the container; show how it loaded (GPU offload, errors)
        print(tag, "llama-server log tail:\n", "\n".join(log.read_text().splitlines()[-25:]))
    res["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    return res


@app.local_entrypoint()
def main(tags: str = ",".join(MODELS), limit: int = 0):
    rows = [json.loads(line) for line in Path("data/test.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = rows[:limit] if limit else rows    # --limit 20: a quick speed check before the full run
    Path("results").mkdir(exist_ok=True)
    tags = tags.split(",")
    calls = evaluate_tag.starmap([(tag, rows) for tag in tags], return_exceptions=True)   # one L4 per tag, in parallel
    for tag, res in zip(tags, calls):
        if isinstance(res, Exception):     # keep the other tags' results
            print(f"{tag} failed: {res}")
            continue
        if limit:
            print(f"{tag}: {limit} rows only, not saved; p50 {res['latency_p50_s']} s, "
                  f"decode {res['decode_tok_s_median']} tok/s, output {res['output_tokens']} tokens")
            continue
        Path(f"results/{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{tag} on {res['hardware']}: accuracy {res['accuracy']:.3f}, macro-F1 {res['macro_f1']:.3f}, "
              f"valid JSON {res['json_valid_rate_unconstrained']:.3f}, p50 {res['latency_p50_s']} s")
