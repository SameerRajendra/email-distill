"""Remove personal identifiers from data/raw.jsonl before any email leaves this PC (Claude CLI, Modal).

Rules run in this order: URLs shrink to scheme+host (tracking tokens in query strings carry PII, and
this goes first so nothing later can split a URL), then your own email addresses, phone numbers, street address and names
(from redact.local.json, git-ignored; copy redact.example.json; addresses before names, or a first name
would eat the front of your own address), then any other email address shrinks to its @domain (sender
domains like ashbyhq.com are the classifier's strongest signal), US-style phone numbers and 6+ digit codes
are masked. Writes data/redacted.jsonl and prints how often each rule fired, so you can sanity-check it.

    python redact.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

URL = ("url", re.compile(r"(https?://[^\s/?#<>\"')\]]+)\S*", re.I), r"\1")
GENERIC = [  # (rule, pattern, replacement)
    ("email", re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)"), r"@\1"),
    ("phone", re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![\w-])"), "[PHONE]"),
    ("number", re.compile(r"\b\d{6,}\b"), "[NUM]"),
    # Tracking and order ids (UPS 1Z..., and the like): letters and digits mixed, 12+ characters.
    ("tracking-id", re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{12,}\b"), "[ID]"),
]
# Names match inside longer tokens too: handles like "SameerRajendra" or "Name_Surname_Resume.pdf" have no
# word boundary between the parts.
OWN = {"emails": "[EMAIL]", "phones": "[PHONE]", "addresses": "[ADDRESS]", "names": "[NAME]"}


def own_rules(path: Path) -> list:
    """Literal, case-insensitive rules for the user's own identifiers, longest first."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    rules = []
    for key, token in OWN.items():
        items = sorted((x for x in cfg.get(key, []) if x.strip()), key=len, reverse=True)
        if items:
            alt = "|".join(re.escape(x) for x in items)
            rules.append((f"own-{key}", re.compile(alt, re.I), token))
    return rules


def redact(text: str, rules: list, counts: Counter) -> str:
    for name, pattern, repl in rules:
        text, n = pattern.subn(repl, text)
        counts[name] += n
    return text


def main():
    local = Path("redact.local.json")
    if not local.exists():
        sys.exit("redact.local.json is missing: copy redact.example.json to it and fill in your own details")
    rules = [URL] + own_rules(local) + GENERIC
    counts = Counter()
    rows = [json.loads(line) for line in Path("data/raw.jsonl").read_text(encoding="utf-8").splitlines()]
    with Path("data/redacted.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            for field in ("from", "subject", "body"):
                r[field] = redact(r[field], rules, counts)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} emails redacted -> data/redacted.jsonl")
    for name, _, _ in rules:
        print(f"  {name:<14}{counts[name]}")


if __name__ == "__main__":
    main()
