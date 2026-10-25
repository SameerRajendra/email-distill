"""Teacher labels: Claude labels the redacted emails through the `claude` CLI (your subscription, not API billing).

20 emails go in per call under short ids (e1..e20) that are mapped back afterwards: long Message-IDs
get garbled by models and a garbled id silently loses its label. Resume-safe: ids already in
data/labels.jsonl are skipped. A batch that fails is retried once, then skipped; rerun to pick it up.

    python label.py
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from schema import FIELDS, KINDS, SYSTEM, format_email

BATCH = 20
MODEL = "sonnet"


def ask_claude(user: str) -> str:
    exe = shutil.which("claude") or shutil.which("claude.cmd")
    if not exe:
        raise SystemExit("claude CLI not found on PATH")
    # ANTHROPIC_API_KEY would make the CLI bill the API instead of the subscription.
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = subprocess.run(
        [exe, "-p", "--output-format", "json", "--model", MODEL, "--tools", "", "--no-session-persistence",
         "--system-prompt", SYSTEM + "\n\nOutput ONLY the JSON object, no prose, no code fences."],
        input=user, capture_output=True, text=True, encoding="utf-8", env=env,
        cwd=tempfile.gettempdir(), timeout=1500)       # a neutral cwd keeps project files out of the context
    data = json.loads(proc.stdout[proc.stdout.index("{"):])
    if data.get("is_error") or proc.returncode:
        raise RuntimeError(str(data.get("result") or proc.stderr)[:300])
    return data["result"]


def label_batch(batch: list[dict]) -> dict[str, dict]:
    """{message id: label} for the emails Claude answered validly."""
    short = {f"e{i}": e["id"] for i, e in enumerate(batch, 1)}
    user = "\n\n".join(format_email(e, f"e{i}") for i, e in enumerate(batch, 1))
    answer = json.loads(re.search(r"\{.*\}", ask_claude(user), re.S).group(0))
    out = {}
    for e in answer["emails"]:
        mid = short.get(str(e.get("id", "")).strip())
        if mid and e.get("kind") in KINDS:
            out[mid] = {k: str(e.get(k, "")).strip() for k in FIELDS[1:]}
    return out


def main():
    emails = [json.loads(line) for line in Path("data/redacted.jsonl").read_text(encoding="utf-8").splitlines()]
    out = Path("data/labels.jsonl")
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [e for e in emails if e["id"] not in done]
    print(f"{len(emails)} emails, {len(done)} labelled, {len(todo)} to do")

    skipped = 0
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            labels = {}
            for attempt in (1, 2):
                try:
                    labels = label_batch(batch)
                except (RuntimeError, ValueError, AttributeError, KeyError, subprocess.TimeoutExpired) as e:
                    print(f"  batch {i // BATCH + 1} attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
                if labels:
                    break
            if not labels:
                skipped += len(batch)
            for mid, label in labels.items():
                f.write(json.dumps({"id": mid, **label}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  batch {i // BATCH + 1}: {len(labels)}/{len(batch)} labelled")
    print(f"done; {skipped} email(s) in failed batches were skipped, rerun to retry them")


if __name__ == "__main__":
    main()
