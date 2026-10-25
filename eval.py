"""Score GGUF models on the held-out test emails against the teacher's labels, served by llama.cpp.

Each model classifies every test email twice: with a JSON-schema constraint (how the tracker runs it;
accuracy, latency and token numbers come from this pass) and unconstrained (to measure how often the
model emits valid JSON on its own). Writes results/<tag>.json.

    python eval.py --model models\\base-Q4_K_M.gguf --tag base-q4 --model models\\email-distill-Q4_K_M.gguf --tag tuned-q4
    python eval.py --url http://127.0.0.1:8080 --tag running-server     (use a server that is already up)
"""
import argparse
import json
import math
import re
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from rapidfuzz import fuzz

from schema import FIELDS, KINDS, VERDICT_SCHEMA

PORT = 8090
SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "co", "company"}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def healthy(url: str) -> bool:
    try:
        return urllib.request.urlopen(url + "/health", timeout=5).status == 200
    except OSError:                      # refused, or 503 while the model loads
        return False


@contextmanager
def serve(exe: str, model: str, log: Path, slots: int = 1):
    with log.open("w") as f:
        proc = subprocess.Popen([exe, "-m", model, "-ngl", "99", "-c", str(4096 * slots), "-fa", "on", "--jinja",
                                 "--port", str(PORT), "-np", str(slots)], stdout=f, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{PORT}"
            deadline = time.time() + 300
            while not healthy(url):
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"llama-server did not start; see {log}")
                time.sleep(2)
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


def classify(url: str, messages: list[dict], constrained: bool) -> tuple[str, float, dict, dict]:
    body = {"messages": messages, "temperature": 0, "max_tokens": 512}
    if constrained:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "result", "strict": True, "schema": VERDICT_SCHEMA}}
    t0 = time.perf_counter()
    res = post(url + "/v1/chat/completions", body)
    return res["choices"][0]["message"]["content"], time.perf_counter() - t0, res["timings"], res["usage"]


def parse(text: str) -> dict | None:
    """The first email's entry if `text` is JSON of the schema's shape, else None."""
    try:
        e = json.loads(text)["emails"][0]
        return e if all(k in e for k in FIELDS) and e["kind"] in KINDS else None
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def norm(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return " ".join(w for w in words if w not in SUFFIXES)


def employer_match(gold: str, pred: str) -> bool:
    g, p = norm(gold), norm(pred)
    return g == p or fuzz.token_set_ratio(g, p) >= 90


def percentile(xs: list[float], q: float) -> float:
    return sorted(xs)[max(0, math.ceil(q * len(xs)) - 1)]


def score(gold: list[dict], pred: list[dict | None]) -> dict:
    cols = KINDS + ["invalid"]
    confusion = {g: {c: 0 for c in cols} for g in KINDS}
    for g, p in zip(gold, pred):
        confusion[g["kind"]][p["kind"] if p else "invalid"] += 1
    per_class = {}
    for k in KINDS:
        tp, support = confusion[k][k], sum(confusion[k].values())
        predicted = sum(confusion[g][k] for g in KINDS)
        prec, rec = tp / predicted if predicted else 0.0, tp / support if support else 0.0
        per_class[k] = {"precision": round(prec, 4), "recall": round(rec, 4), "support": support,
                        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}
    seen = [c for c in per_class.values() if c["support"]]
    emp = [employer_match(g["company"], p["company"] if p else "") for g, p in zip(gold, pred)
           if g["kind"] != "other" and norm(g["company"])]
    correct = sum(confusion[k][k] for k in KINDS)
    return {"n": len(gold), "accuracy": round(correct / len(gold), 4),
            "macro_f1": round(statistics.mean(c["f1"] for c in seen), 4), "per_class": per_class,
            "employer_match": {"rate": round(sum(emp) / len(emp), 4) if emp else None, "n": len(emp)},
            "confusion": confusion}


def evaluate(url: str, tag: str, model: str | None, rows: list[dict], slots: int = 1) -> dict:
    gold = [json.loads(r["messages"][2]["content"])["emails"][0] for r in rows]
    prompts = [r["messages"][:2] for r in rows]
    runs = [classify(url, m, True) for m in prompts]
    # Latency is timed one request at a time (above); the validity-only pass may use parallel server slots.
    with ThreadPoolExecutor(slots) as pool:
        valid = list(pool.map(lambda m: parse(classify(url, m, False)[0]) is not None, prompts))
    lat = [r[1] for r in runs]
    decode = [r[2]["predicted_per_second"] for r in runs]
    prefill = [r[2]["prompt_per_second"] for r in runs]
    return {"tag": tag, "model": model, **score(gold, [parse(r[0]) for r in runs]),
            "json_valid_rate_unconstrained": round(sum(valid) / len(valid), 4),
            "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(percentile(lat, 0.95), 3),
            "decode_tok_s_median": round(statistics.median(decode), 1),
            "prefill_tok_s_median": round(statistics.median(prefill), 1),
            "prompt_tokens": sum(r[3]["prompt_tokens"] for r in runs),
            "output_tokens": sum(r[3]["completion_tokens"] for r in runs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="GGUF path; repeat, one --tag each")
    ap.add_argument("--tag", action="append", required=True, help="name for the results file")
    ap.add_argument("--llama-server", default=r"D:\LLM models\llama.cpp\llama-server.exe")
    ap.add_argument("--url", help="use this running llama-server instead of starting one (one --tag)")
    a = ap.parse_args()
    if a.url and (len(a.tag) != 1 or a.model):
        ap.error("--url takes exactly one --tag and no --model")
    if not a.url and len(a.model) != len(a.tag):
        ap.error("give one --tag per --model")

    rows = [json.loads(line) for line in Path("data/test.jsonl").read_text(encoding="utf-8").splitlines()]
    Path("results").mkdir(exist_ok=True)
    for tag, model in zip(a.tag, a.model or [None]):
        if a.url:
            res = evaluate(a.url.rstrip("/"), tag, None, rows)
        else:
            with serve(a.llama_server, model, Path(f"results/{tag}.server.log")) as url:
                res = evaluate(url, tag, model, rows)
        Path(f"results/{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{tag}: accuracy {res['accuracy']:.3f}, macro-F1 {res['macro_f1']:.3f}, "
              f"valid JSON {res['json_valid_rate_unconstrained']:.3f}, p50 {res['latency_p50_s']} s")


if __name__ == "__main__":
    main()
