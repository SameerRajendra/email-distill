"""Pull application-like emails from Thunderbird into data/raw.jsonl.

Uses the same Thunderbird add-on API and the same sender/subject prefilters as the tracker, so the
training pool looks like the tracker's real input. Resume-safe: ids already in the file are skipped and
each email is appended as soon as its body is fetched.

    python export.py [--days 365]
"""
import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Prefilters copied from tracker_sync.py.
ATS_SENDERS = re.compile(r"ashbyhq|greenhouse|lever\.co|myworkday|workday|workablemail|tsenta|icims|smartrecruiters|"
                         r"jobvite|bamboohr|rippling|pinpoint|breezy|recruitee|teamtailor|dover|gem\.com|"
                         r"successfactors|taleo|oraclecloud|eightfold|avature|jazzhr|recruiting|talent|careers|hiring|"
                         r"no-?reply", re.I)
SUBJECT_HINT = re.compile(r"appl(y|ied|ication)|thank(s| you) for (your )?(applying|interest|applying)|received|"
                          r"update on|update from|your candidacy|next steps|interview|unfortunately|decision|"
                          r"moving forward|regarding your|status of your|assessment|coding challenge|schedul", re.I)
NOT_APPLICATION = re.compile(r"jobalerts-noreply@linkedin|jobs-noreply@linkedin|invitations@linkedin|"
                             r"newsletter|digest|webinar|job alert|jobs you may be interested|is hiring|"
                             r"security code|verification code|one-time|password", re.I)
BODY_CHARS = 3000      # schema.format_email keeps 1500; the margin lets redaction shorten URLs first


def tb_call(name: str, args: dict, timeout: int = 90):
    """JSON-RPC to the Thunderbird MCP add-on on 127.0.0.1 (port + bearer token in connection.json)."""
    conn = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "thunderbird-mcp" / "connection.json"
    c = json.loads(conn.read_text(encoding="utf-8"))
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{c['port']}/", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {c['token']}"})
    res = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if "error" in res:
        raise RuntimeError(str(res["error"])[:200])
    return json.loads(res["result"]["content"][0]["text"])


def search_headers(since: str) -> list[dict]:
    heads, offset, size = [], 0, 100
    while True:
        try:
            page = tb_call("searchMessages", {"query": "", "startDate": since, "maxResults": size, "offset": offset})
        except RuntimeError as e:
            # Thunderbird answered "out of memory" to a 200-message page once: retry the page smaller.
            if size <= 10:
                raise
            size = max(10, size // 2)
            print(f"thunderbird refused a page ({str(e)[:60]}); retrying with {size} per page")
            continue
        items = page.get("messages", []) if isinstance(page, dict) else page
        heads += items
        if not (isinstance(page, dict) and page.get("hasMore")) or not items:
            return heads
        offset += len(items)


def looks_like_application(h: dict) -> bool:
    frm, subj = h.get("author", ""), h.get("subject", "")
    return bool(h.get("id")) and not NOT_APPLICATION.search(frm + " " + subj) \
        and bool(ATS_SENDERS.search(frm) or SUBJECT_HINT.search(subj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    out = Path("data/raw.jsonl")
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()

    now = datetime.now(timezone.utc)
    heads = search_headers((now - timedelta(days=a.days)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    todo = []
    for h in heads:
        try:
            when = datetime.fromisoformat(h["date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if when < now + timedelta(days=2) and h["id"] not in done and looks_like_application(h):
            todo.append((when, h))                   # bogus future dates exist, hence the cutoff
    todo.sort(key=lambda t: t[0])
    print(f"{len(heads)} messages in {a.days} days, {len(todo)} new application-like, {len(done)} already exported")

    failed = 0
    with out.open("a", encoding="utf-8") as f:
        for n, (when, h) in enumerate(todo, 1):
            try:
                body = tb_call("getMessage", {"messageId": h["id"], "folderPath": h["folderPath"],
                                              "bodyFormat": "text"}).get("body", "")
            except Exception:
                failed += 1                          # not written, so the next run retries it
                continue
            f.write(json.dumps({"id": h["id"], "date": when.isoformat(), "from": h.get("author", ""),
                                "subject": h.get("subject", ""),
                                "body": re.sub(r"\s+", " ", body or "").strip()[:BODY_CHARS]},
                               ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"done; {failed} body fetch(es) failed (rerun to retry); output {out}")


if __name__ == "__main__":
    main()
