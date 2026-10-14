"""
retrieval_eval.py - BEFORE / AFTER test: vector-only RAG vs hybrid RAG.

    python retrieval_eval.py                       (uses gemini-3.6-flash as the judge)
    python retrieval_eval.py gemini-3.5-flash-lite (pick another judge model)

For each role's 10 tasks, both methods retrieve their top-3 case studies.
All retrieved cases for a task are pooled, SHUFFLED and shown WITHOUT saying which method
found them (a blind test - avoids the judge favouring one method), and an LLM judge rates
each one:  2 = directly relevant evidence,  1 = partly relevant,  0 = not relevant.

Cost: 1 Gemini request per role = 5 requests in total.
Output: outputs/retrieval_eval.csv + a summary table for the report.
"""
import json
import random
import sys
import time

import pandas as pd

import config
from evaluate import _api_key
from retrieval import search
from tools import _case_collection, _tasks, select_top_tasks

K = 3
JUDGE_PROMPT = """You are judging a search engine for workplace AI case studies.
For each TASK below you get candidate case studies (company, title, excerpt).
Rate how useful each candidate is as EVIDENCE about AI being used for that specific task:
  2 = directly relevant (AI used for this task or a very close equivalent)
  1 = partly relevant (same domain or a related activity)
  0 = not relevant
Return ONLY JSON: a list of objects {"task": <task number>, "cand": <candidate letter>, "score": 0|1|2}.
"""


def build_items(role, tasks, col):
    items = []
    for n, (_, t) in enumerate(tasks.iterrows(), start=1):
        query = f"AI for {t['task']}"
        found = {}
        for mode in ("vector", "hybrid"):
            for rank, h in enumerate(search(col, query, K, mode=mode), start=1):
                cid = h["meta"]["case_id"]
                found.setdefault(cid, {"meta": h["meta"], "doc": h["doc"], "modes": {}})
                found[cid]["modes"][mode] = rank
        cands = list(found.values())
        random.Random(f"{role}-{n}").shuffle(cands)          # blind + reproducible order
        for letter, c in zip("ABCDEFGH", cands):
            c["letter"] = letter
        items.append({"n": n, "task": t["task"], "task_id": t["task_id"], "cands": cands})
    return items


def judge(items, client, models):
    """Ask the LLM judge for scores. Tries each model in turn: if one is overloaded (503)
    or out of daily quota, it moves on to the next one."""
    from google.genai import types
    lines = [JUDGE_PROMPT]
    for it in items:
        lines.append(f"\nTASK {it['n']}: {it['task']}")
        for c in it["cands"]:
            m = c["meta"]
            lines.append(f"  {c['letter']}) {m['company']} - {m['title']}: {c['doc'][:200]}")
    prompt = "\n".join(lines)
    for model in models:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                data = json.loads(resp.text)
                return {(int(r["task"]), str(r["cand"])): int(r["score"]) for r in data}, model
            except Exception as e:
                msg = str(e).lower().replace("_", "")
                if "perday" in msg:
                    print(f"   daily quota used up for {model} - trying the next model")
                    break
                print(f"   {model} busy or failed ({e.__class__.__name__}) - retry {attempt + 1}/3")
                time.sleep(15 * (attempt + 1))
        print(f"   switching away from {model}")
    raise RuntimeError("All judge models failed (Google servers busy). Wait 10-15 minutes and run again - "
                       "finished roles are saved and will be skipped.")


def main():
    from google import genai
    first = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.6-flash"
    models = [first] + [m for m in config.PROVIDERS[config.DEFAULT_PROVIDER]["models"] if m != first]
    key = _api_key(config.DEFAULT_PROVIDER)
    if not key:
        sys.exit("No API key found (.streamlit/secrets.toml or GEMINI_API_KEY).")
    client, col, tasks = genai.Client(api_key=key), _case_collection(), _tasks()
    print("Preparing keyword index (first time takes ~30 s)...")
    rows = []
    for role in config.ROLES:
        sub = select_top_tasks(tasks[tasks["role"] == role])
        items = build_items(role, sub, col)
        cache = config.OUTPUT_DIR / f"retrieval_judge_{role.replace(' ', '_')}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            scores = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in saved["scores"].items()}
            print(f"Skip {role} - already judged (saved)")
        else:
            print(f"Judging {role} ({sum(len(i['cands']) for i in items)} candidates) ...")
            scores, used = judge(items, client, models)
            config.OUTPUT_DIR.mkdir(exist_ok=True)
            cache.write_text(json.dumps({"model": used, "scores": {f"{k[0]}|{k[1]}": v for k, v in scores.items()}}))
            print(f"   done with {used}")
        for it in items:
            for mode in ("vector", "hybrid"):
                picked = [c for c in it["cands"] if mode in c["modes"]]
                vals = [scores.get((it["n"], c["letter"]), 0) for c in picked]
                rows.append({"role": role, "task_id": it["task_id"], "mode": mode,
                             "mean_relevance": sum(vals) / K,
                             "precision_at_3": sum(v >= 1 for v in vals) / K,
                             "has_direct_hit": int(any(v == 2 for v in vals))})
        time.sleep(config.GEMINI_MIN_SECONDS)
    df = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(config.OUTPUT_DIR / "retrieval_eval.csv", index=False)
    summary = df.groupby("mode")[["mean_relevance", "precision_at_3", "has_direct_hit"]].mean().round(3)
    by_role = df.pivot_table(index="role", columns="mode", values="precision_at_3").round(2)
    pd.set_option("display.width", 200)
    print("\nOVERALL (50 tasks, top-3 each)\n", summary.to_string())
    print("\nPRECISION@3 BY ROLE\n", by_role.to_string())
    print("\nmean_relevance: 0-2 scale | precision_at_3: share of top-3 hits rated relevant (>=1) | "
          "has_direct_hit: share of tasks with at least one directly relevant case")
    print(f"Saved {config.OUTPUT_DIR / 'retrieval_eval.csv'}")


if __name__ == "__main__":
    main()
