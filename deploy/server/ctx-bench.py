#!/usr/bin/env python3
"""Context-scaling tok/s bench. Measures decode tok/s as prompt size grows.
Usage: ctx-bench.py [port] [label]
"""
import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
LABEL = sys.argv[2] if len(sys.argv) > 2 else "mode"

FILLER = (
    "The history of computing is a long and interesting story. From the earliest "
    "mechanical calculators built by Blaise Pascal in the seventeenth century "
    "through the analytical engine designed by Charles Babbage the field evolved "
    "through distinct eras. Early electronic computers used vacuum tubes and "
    "punch cards. The transistor revolution brought reliability and "
    "miniaturization. By 1970 integrated circuits enabled personal computing. "
    "The 1980s saw the rise of the IBM PC and Apple Macintosh. The 1990s "
    "brought the World Wide Web and graphical browsers. The 2000s saw mobile "
    "computing explode. Cloud computing arrived in the 2010s followed by deep "
    "learning and large language models. Today transformers and attention "
    "based architectures dominate natural language processing. "
)

# Test progressively larger prompts. Each FILLER copy is ~150 tokens.
SIZES = [1, 50, 200, 500, 1000, 1500]

print(f"=== {LABEL} ===")
print(f"  {'ptok':>6}  {'decode':>9}  {'pp':>8}  {'total_ms':>10}  draft")
print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*22}")

for count in SIZES:
    user_msg = "Summarize the key milestones in this text in 100 words: " + FILLER * count
    body = {
        "model": "Qwen/Qwen3.6-27B",
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": 256,
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            d = json.loads(resp.read())
    except Exception as e:
        print(f"  count={count}  ERROR: {e}")
        continue
    u = d.get("usage", {})
    t = d.get("timings", {})
    ptok = u.get("prompt_tokens", 0)
    decode = round(t.get("predicted_per_second", 0), 1)
    pp = round(t.get("prompt_per_second", 0), 1)
    total_ms = round(t.get("prompt_ms", 0) + t.get("predicted_ms", 0), 0)
    draft_n = t.get("draft_n", 0)
    draft_acc = t.get("draft_n_accepted", 0)
    draft_pct = f"{round(100*draft_acc/draft_n,1)}%" if draft_n else "n/a"
    print(f"  {ptok:>6}  {decode:>7} t/s  {pp:>6} t/s  {int(total_ms):>10}  draft={draft_n}/acc={draft_acc} ({draft_pct})")
