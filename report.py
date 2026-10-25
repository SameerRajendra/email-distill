"""Aggregate results/*.json into results/REPORT.md: one comparison table, then a confusion matrix per model.

API prices are arguments, never hardcoded: cost per 1k emails is the measured prompt/output tokens per
email (counted by the local model's tokenizer, so an approximation of what an API would bill) times your prices.

    python report.py --in-usd-per-mtok 3 --out-usd-per-mtok 15
"""
import argparse
import json
from pathlib import Path

from schema import KINDS


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-usd-per-mtok", type=float, required=True, help="API price per million input tokens")
    ap.add_argument("--out-usd-per-mtok", type=float, required=True, help="API price per million output tokens")
    a = ap.parse_args()
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path("results").glob("*.json"))]
    if not results:
        raise SystemExit("no results/*.json: run eval.py first")

    def cost_per_1k(r):
        return 1000 * (r["prompt_tokens"] * a.in_usd_per_mtok + r["output_tokens"] * a.out_usd_per_mtok) / 1e6 / r["n"]

    md = ["# Results", "",
          f"Agreement with the teacher's labels on {results[0]['n']} held-out emails (the newest 20% of the mail).", ""]
    md += table(["Model", "Accuracy", "Macro-F1", "Employer match", "Valid JSON (unconstrained)", "Latency p50 / p95 (s)",
                 "Decode tok/s", "Prefill tok/s", f"API-equivalent $ per 1k emails (${a.in_usd_per_mtok:g} in, ${a.out_usd_per_mtok:g} out per Mtok)"],
                [[r["tag"], pct(r["accuracy"]), f"{r['macro_f1']:.3f}",
                  f"{pct(r['employer_match']['rate'])} (n={r['employer_match']['n']})",
                  pct(r["json_valid_rate_unconstrained"]), f"{r['latency_p50_s']} / {r['latency_p95_s']}",
                  r["decode_tok_s_median"], r["prefill_tok_s_median"], f"{cost_per_1k(r):.3f}"] for r in results])
    for r in results:
        md += [f"## {r['tag']}", "", "Confusion matrix (rows: teacher, columns: model)", ""]
        md += table(["teacher \\ model"] + KINDS + ["invalid"],
                    [[g] + [r["confusion"][g][c] for c in KINDS + ["invalid"]] for g in KINDS])
        md += table(["class", "support", "precision", "recall", "F1"],
                    [[k, c["support"], f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1']:.3f}"]
                     for k, c in r["per_class"].items()])
    Path("results/REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
