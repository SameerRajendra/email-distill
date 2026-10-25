"""Join redacted emails with the teacher's labels and write chat-format train/test files.

The split is by time (oldest 80% train, newest 20% test), not random: the tracker only ever sees mail
newer than anything it learned from, and templated ATS emails repeat their wording, so a random split
would put near-duplicates on both sides and flatter the score.

    python prepare.py
"""
import json
from collections import Counter
from pathlib import Path

from schema import FIELDS, STUDENT_SYSTEM, format_email

TRAIN_SHARE = 0.8


def jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def example(email: dict, label: dict) -> dict:
    answer = {"emails": [{k: "e1" if k == "id" else label[k] for k in FIELDS}]}
    return {"messages": [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": format_email(email)},
        {"role": "assistant", "content": json.dumps(answer, separators=(",", ":"), ensure_ascii=False)}]}


def main():
    labels = {r["id"]: r for r in jsonl("data/labels.jsonl")}
    every = jsonl("data/redacted.jsonl")
    emails = sorted((e for e in every if e["id"] in labels), key=lambda e: e["date"])
    print(f"{len(every) - len(emails)} email(s) without a label dropped")
    cut = int(len(emails) * TRAIN_SHARE)
    for name, part in (("train", emails[:cut]), ("test", emails[cut:])):
        with Path(f"data/{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(example(e, labels[e["id"]]), ensure_ascii=False) + "\n" for e in part)
        counts = Counter(labels[e["id"]]["kind"] for e in part)
        print(f"{name}: {len(part)} emails, {part[0]['date'][:10]} .. {part[-1]['date'][:10]}")
        print("  " + ", ".join(f"{k} {n}" for k, n in counts.most_common()))


if __name__ == "__main__":
    main()
