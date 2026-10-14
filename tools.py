"""
tools.py - The agent's five TOOLS (its "hands"). Each is a plain Python function.

  1. find_occupation      job title  -> official O*NET occupation
  2. get_tasks            occupation -> its most important tasks
  3. get_ai_usage         tasks      -> how much people already use AI on them (Economic Index)
  4. search_case_studies  task       -> real companies that used AI on it (RAG over ZenML)
  5. score_job_structure  scores     -> automate / augment / human-only verdicts + job shares

Tools 1, 2, 3 and 5 are deterministic. Tool 4 is hybrid RAG: embeddings + cosine similarity
AND BM25 keyword search, fused with Reciprocal Rank Fusion (retrieval.py).
The LLM decides WHICH tool to call and WHEN (the ReAct loop in agent.py).
"""
import threading
from functools import lru_cache

import pandas as pd

import config

LEVELS = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# Data loading (cached so it happens once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _tasks() -> pd.DataFrame:
    if not config.TASKS_CSV.exists():
        raise FileNotFoundError("data/tasks.csv not found. Run:  python data_prep.py")
    return pd.read_csv(config.TASKS_CSV, dtype={"task_id": str})


_collection = None


_collection_lock = threading.Lock()


def _case_collection(embedding_function=None):
    """Open the vector database once per server process. The lock stops two users (or two
    parallel tool calls) from opening/building it at the same moment."""
    global _collection
    with _collection_lock:
        if _collection is None:
            import chromadb
            from data_prep import ensure_case_index, get_embedding_function
            ensure_case_index()          # builds a small index from cases.csv if none exists (cloud)
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            _collection = client.get_collection(
                "case_studies", embedding_function=embedding_function or get_embedding_function()
            )
    return _collection


def select_top_tasks(role_tasks: pd.DataFrame, top_n: int = config.TOP_N_TASKS) -> pd.DataFrame:
    """The role's most important tasks. Used by BOTH the agent and the evaluation sheets,
    so they always see the same tasks. If O*NET has no importance ratings for a role,
    fall back to its 'Core' tasks in O*NET order with a neutral importance of 3.0."""
    rated = role_tasks.dropna(subset=["importance"])
    if len(rated) >= top_n:
        return rated.sort_values(["importance", "task_id"], ascending=[False, True]).head(top_n)
    out = role_tasks.copy()
    out["_core"] = (out["task_type"].astype(str).str.lower() == "core").astype(int)
    out["_id"] = pd.to_numeric(out["task_id"], errors="coerce")
    out = out.sort_values(["_core", "_id"], ascending=[False, True]).head(top_n)
    out["importance"] = out["importance"].fillna(3.0)
    return out.drop(columns=["_core", "_id"])


# ---------------------------------------------------------------------------
# Tool 1
# ---------------------------------------------------------------------------
def find_occupation(job: str) -> dict:
    """Match the user's job title to one of the roles in scope."""
    text = job.lower().strip()
    best, best_score = None, 0
    for role, info in config.ROLES.items():
        names = [role.lower(), info["onet_title"].lower()] + info["aliases"]
        score = max((len(n) for n in names if n in text or text in n), default=0)
        if score > best_score:
            best, best_score = role, score
    if best is None:
        return {
            "matched": False,
            "message": "This job is outside the project scope.",
            "roles_in_scope": list(config.ROLES.keys()),
        }
    info = config.ROLES[best]
    return {"matched": True, "role": best, "onet_code": info["onet_code"], "onet_title": info["onet_title"]}


# ---------------------------------------------------------------------------
# Tool 2
# ---------------------------------------------------------------------------
def get_tasks(onet_code: str, top_n: int = config.TOP_N_TASKS) -> dict:
    """Return the occupation's most important O*NET tasks (importance is rated 1-5)."""
    df = _tasks()
    sub = select_top_tasks(df[df["onet_code"] == onet_code], top_n)
    if sub.empty:
        return {"error": f"No tasks found for {onet_code}"}
    return {
        "onet_code": onet_code,
        "onet_title": sub["onet_title"].iloc[0],
        "tasks": [
            {"task_id": str(r.task_id), "task": r.task,
             "importance": None if pd.isna(r.importance) else round(float(r.importance), 2)}
            for r in sub.itertuples()
        ],
    }


# ---------------------------------------------------------------------------
# Tool 3
# ---------------------------------------------------------------------------
def get_ai_usage(task_ids: list) -> dict:
    """For each task: share of AI conversations involving it, and whether that use
    is mostly automation (AI does it) or augmentation (AI helps a human)."""
    df = _tasks()
    out = []
    for tid in task_ids:
        row = df[df["task_id"] == str(tid)]
        if row.empty:
            out.append({"task_id": str(tid), "error": "unknown task_id"})
            continue
        r = row.iloc[0]
        usage = float(r["ai_usage_pct"]) if not pd.isna(r["ai_usage_pct"]) else 0.0
        item = {
            "task_id": str(tid),
            "ai_usage_pct": round(usage, 4),
            "has_usage_data": usage > 0,
        }
        if not pd.isna(r.get("automation_share", float("nan"))):
            item["automation_share"] = float(r["automation_share"])
            item["augmentation_share"] = float(r["augmentation_share"])
        out.append(item)
    return {
        "source": "Anthropic Economic Index (release 2025-03-27)",
        "note": "ai_usage_pct = % of sampled Claude conversations involving the task; 0 means no recorded use.",
        "results": out,
    }


# ---------------------------------------------------------------------------
# Tool 4 (RAG)
# ---------------------------------------------------------------------------
def _search_one(col, query: str, k: int):
    from retrieval import search
    hits = []
    for h in search(col, query, k):
        meta = h["meta"]
        item = {"company": meta["company"], "title": meta["title"], "industry": meta["industry"],
                "year": meta["year"], "url": meta["url"], "excerpt": h["doc"][:400]}
        if "similarity" in h:
            item["similarity"] = h["similarity"]          # cosine similarity (vector match)
        if "found_by" in h:
            item["found_by"] = h["found_by"]              # vector, keyword or both
        hits.append(item)
    return hits


def search_case_studies(queries=None, query: str = None, k: int = config.TOP_K) -> dict:
    """RAG retrieval: for each query, the top-k case-study chunks by HYBRID search
    (cosine similarity + BM25 keywords, fused with RRF; see retrieval.py).
    Accepts several queries at once so the agent needs fewer round-trips."""
    if queries is None:
        queries = [query] if query else []
    if isinstance(queries, str):
        queries = [queries]
    col = _case_collection()
    return {"results": [{"query": q, "hits": _search_one(col, q, k)} for q in queries[:8]],
            "note": "Company results in case studies are usually vendor-reported."}


# ---------------------------------------------------------------------------
# Tool 5
# ---------------------------------------------------------------------------
def decide_verdict(routine: str, information: str, judgement: str, human_touch: str) -> str:
    """The fixed decision rule from the project design (same input -> same verdict)."""
    ro, inf, ju, hu = (LEVELS.get(str(x).lower(), 1) for x in (routine, information, judgement, human_touch))
    if hu == 2:
        return "Human-only"
    if ju == 2 and inf < 2:
        return "Human-only"
    if ro == 2 and inf == 2 and ju == 0:
        return "Automate"
    if inf >= 1:
        return "Augment"
    return "Human-only"


def score_job_structure(tasks: list) -> dict:
    """tasks: list of {task_id, routine, information, judgement, human_touch} (Low/Medium/High).
    Applies the decision rule and returns importance-weighted shares of the job."""
    df = _tasks()
    results, totals = [], {"Automate": 0.0, "Augment": 0.0, "Human-only": 0.0}
    for t in tasks:
        verdict = decide_verdict(t.get("routine"), t.get("information"),
                                 t.get("judgement"), t.get("human_touch"))
        row = df[df["task_id"] == str(t.get("task_id"))]
        weight = float(row["importance"].iloc[0]) if not row.empty and not pd.isna(row["importance"].iloc[0]) else 3.0
        totals[verdict] += weight
        results.append({"task_id": str(t.get("task_id")), "verdict": verdict, "importance": weight})
    total = sum(totals.values()) or 1.0
    shares = {k: round(100 * v / total, 1) for k, v in totals.items()}
    return {"verdicts": results, "shares_pct": shares,
            "method": "importance-weighted share of tasks per verdict"}


# ---------------------------------------------------------------------------
# Tool descriptions given to Claude (JSON schema)
# ---------------------------------------------------------------------------
LEVEL = {"type": "string", "enum": ["Low", "Medium", "High"]}

TOOL_SPECS = [
    {
        "name": "find_occupation",
        "description": "Map a job title or job description to an official O*NET occupation in the project's scope.",
        "input_schema": {"type": "object", "properties": {"job": {"type": "string"}}, "required": ["job"]},
    },
    {
        "name": "get_tasks",
        "description": "Get the most important O*NET tasks (importance 1-5) for an occupation code.",
        "input_schema": {"type": "object", "properties": {"onet_code": {"type": "string"}}, "required": ["onet_code"]},
    },
    {
        "name": "get_ai_usage",
        "description": "For a list of task_ids, get real-world AI usage from the Anthropic Economic Index, "
                       "including automation vs augmentation share where available. Pass ALL task_ids in one call.",
        "input_schema": {"type": "object",
                         "properties": {"task_ids": {"type": "array", "items": {"type": "string"}}},
                         "required": ["task_ids"]},
    },
    {
        "name": "search_case_studies",
        "description": "Semantic search (RAG) over 2,000+ real company LLM case studies from the ZenML LLMOps "
                       "Database. Pass 4-6 short queries at once, one per task likely affected by AI, e.g. "
                       "['AI resume screening recruiting', 'chatbot answering HR policy questions']. "
                       "Returns, per query: company, title, url, similarity, excerpt.",
        "input_schema": {"type": "object",
                         "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                         "required": ["queries"]},
    },
    {
        "name": "score_job_structure",
        "description": "Apply the fixed decision rule to your Low/Medium/High scores and compute the job's "
                       "automate / augment / human-only shares. Call once with ALL tasks.",
        "input_schema": {
            "type": "object",
            "properties": {"tasks": {"type": "array", "items": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}, "routine": LEVEL, "information": LEVEL,
                               "judgement": LEVEL, "human_touch": LEVEL},
                "required": ["task_id", "routine", "information", "judgement", "human_touch"]}}},
            "required": ["tasks"],
        },
    },
    {
        "name": "submit_analysis",
        "description": "FINAL STEP. Submit the finished analysis. Call exactly once, after score_job_structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "short_task": {"type": "string", "description": "Task in under 10 words"},
                        "routine": LEVEL, "information": LEVEL, "judgement": LEVEL, "human_touch": LEVEL,
                        "why": {"type": "string", "description": "One sentence reason"},
                        "evidence": {"type": "string", "description": "Named source: company case, Economic Index figure, or 'No evidence found'"},
                        "evidence_url": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    },
                    "required": ["task_id", "short_task", "routine", "information", "judgement",
                                 "human_touch", "why", "evidence", "confidence"]}},
                "role_shift_summary": {"type": "string", "description": "3 lines: how the role shifts"},
                "skills_to_build": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tasks", "role_shift_summary", "skills_to_build"],
        },
    },
]

TOOL_FUNCTIONS = {
    "find_occupation": find_occupation,
    "get_tasks": get_tasks,
    "get_ai_usage": get_ai_usage,
    "search_case_studies": search_case_studies,
    "score_job_structure": score_job_structure,
}
