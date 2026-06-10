#!/usr/bin/env python3
"""
Run mem0 against the same LongMemEval stratified 48-item subset that
nizo-memory v3 was scored on. Local-only — Ollama qwen2.5:14b for the
LLM, nomic-embed-text for embeddings, in-memory Qdrant for vectors.

Output JSON shaped like the nizo LongMemEvalRunner report so we can
diff per-item.
"""
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from mem0 import Memory
from collections import defaultdict
import requests

DATASET = '/Users/kislaysinha/data/longmemeval/oracle-stratified-48.jsonl'
OUT     = '/Users/kislaysinha/claude_projects/nizo-memory/bench/lme-oracle-48-mem0.json'
OLLAMA  = 'http://localhost:11434'
MODEL   = 'qwen2.5:14b'
EMBED   = 'nomic-embed-text'

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODEL,
            "ollama_base_url": OLLAMA,
            "temperature": 0.1,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": EMBED,
            "ollama_base_url": OLLAMA,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "lme_bench",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
}

# Fall back to chromadb in-memory if qdrant isn't running
def make_memory():
    try:
        return Memory.from_config(config)
    except Exception as e:
        print(f"qdrant failed ({e}), falling back to chromadb in-memory...", file=sys.stderr)
        config["vector_store"] = {
            "provider": "chroma",
            "config": {
                "collection_name": "lme_bench",
                "path": "/tmp/mem0_chroma",
            },
        }
        return Memory.from_config(config)

ANSWER_PROMPT = """You are answering a user's question using ONLY the memory context below.
If the context doesn't contain the answer, say "I don't know."
Be concise — one or two sentences. No preamble.

MEMORY CONTEXT:
{ctx}

QUESTION: {q}
ANSWER:"""

JUDGE_PROMPT = """You are grading whether a model's answer is correct given the ground truth.
Reply with a single token: YES if the answer is correct (captures the same facts,
even if worded differently), NO if it's wrong, missing, or says "I don't know"
when the ground truth is specific.

QUESTION: {q}
GROUND TRUTH: {gt}
MODEL ANSWER: {ans}
GRADE (YES/NO):"""

def ollama_chat(prompt, temp=0.0):
    """Call Ollama directly so the answerer/judge logic matches our nizo bench."""
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp, "num_ctx": 8192},
    }, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def run():
    items = [json.loads(l) for l in open(DATASET)]
    print(f"Loaded {len(items)} items", file=sys.stderr)

    mem = make_memory()
    results = []
    per_type = defaultdict(lambda: [0, 0])  # total, correct
    t_start = time.time()

    for idx, item in enumerate(items, 1):
        qid = item['question_id']
        uid = f"mem0-bench-{qid}"
        question = item['question']
        gt = item['answer']
        qtype = item['question_type']

        # 1. Ingest sessions. mem0 v2 add() expects messages as a LIST and
        #    runs LLM extraction per call; sending a whole session at once
        #    blows past qwen2.5:14b's 8K context, so we chunk pair-by-pair
        #    (user + assistant). This matches how mem0 is typically called
        #    in production (per-turn or per-pair).
        t_ingest = time.time()
        for session in item['sessions']:
            for k in range(0, len(session), 2):
                pair = session[k:k+2]
                try:
                    mem.add(messages=pair, user_id=uid)
                except Exception as e:
                    print(f"  ingest err on {qid} turn {k}: {str(e)[:80]}", file=sys.stderr)
        ingest_ms = int((time.time() - t_ingest) * 1000)

        # 2. Recall via mem0 search (v2 API uses filters=)
        t_recall = time.time()
        try:
            recalled = mem.search(query=question, filters={'user_id': uid}, limit=10)
        except Exception as e:
            print(f"  search err on {qid}: {e}", file=sys.stderr)
            recalled = {'results': []}
        recall_ms = int((time.time() - t_recall) * 1000)

        # mem0 returns {'results': [{'memory': '...', ...}, ...]}
        rec_items = recalled.get('results', []) if isinstance(recalled, dict) else recalled or []
        rec_content = [r.get('memory', str(r))[:300] for r in rec_items]
        ctx = '\n'.join(f'- {c}' for c in rec_content) or '(no relevant memory)'

        # 3. Answer
        try:
            predicted = ollama_chat(ANSWER_PROMPT.format(ctx=ctx, q=question))
        except Exception as e:
            predicted = f"(answerer failed: {e})"

        # 4. Judge
        try:
            verdict = ollama_chat(JUDGE_PROMPT.format(q=question, gt=gt, ans=predicted))
            correct = verdict.strip().upper().startswith('YES')
        except Exception:
            correct = False

        results.append({
            'questionId': qid,
            'questionType': qtype,
            'question': question,
            'predicted': predicted,
            'groundTruth': gt,
            'correct': correct,
            'ingestMs': ingest_ms,
            'recallMs': recall_ms,
            'recalledItems': len(rec_items),
            'recalledContent': rec_content,
        })
        per_type[qtype][0] += 1
        if correct: per_type[qtype][1] += 1

        # 5. Wipe so next user is fresh
        try:
            mem.delete_all(filters={'user_id': uid})
        except Exception:
            pass

        elapsed = time.time() - t_start
        print(f"  [{idx:>2}/{len(items)}] {qid[:25]:<25} {qtype:<28} "
              f"correct={correct} recalled={len(rec_items):>2}  "
              f"({elapsed:.0f}s elapsed, ~{elapsed/idx*len(items):.0f}s ETA)",
              file=sys.stderr)

    total = len(results)
    correct_n = sum(1 for r in results if r['correct'])
    report = {
        'total': total,
        'correct': correct_n,
        'accuracy': correct_n / total if total else 0,
        'byType': {k: {'total': v[0], 'correct': v[1]} for k, v in per_type.items()},
        'items': results,
    }
    with open(OUT, 'w') as fp:
        json.dump(report, fp, indent=2)
    print(f"\n=== mem0 result ===", file=sys.stderr)
    print(f"  Overall: {correct_n}/{total} = {correct_n/total*100:.1f}%", file=sys.stderr)
    for k, v in per_type.items():
        print(f"  {k:<32} {v[1]}/{v[0]} = {v[1]/v[0]*100 if v[0] else 0:.1f}%", file=sys.stderr)
    print(f"  Elapsed: {time.time()-t_start:.0f}s, written to {OUT}", file=sys.stderr)


if __name__ == '__main__':
    run()
