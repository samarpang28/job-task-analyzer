"""
data_prep.py - STEP 1. Run this ONCE before using the app.

    python data_prep.py

It does four things:
  1. Downloads O*NET (what tasks each job has + how important each task is)
  2. Downloads the Anthropic Economic Index (how much people use AI on each task)
  3. Builds data/tasks.csv for the 5 roles in config.py
  4. Downloads the ZenML case studies and builds the RAG vector database
     (chunking -> embeddings -> Chroma), exactly as taught in Sessions 7-8.

It takes about 5-15 minutes the first time (mostly step 4).
If a download fails, the message tells you how to download the file by hand.
"""
import io
import re
import sys
import zipfile

import pandas as pd
import requests

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (student project: Job Task Impact Analyzer)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_task(text: str) -> str:
    """Make task text comparable across datasets: lowercase, single spaces, no final full stop."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".").strip()


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """Split text into chunks of `size` characters; each chunk repeats the last
    `overlap` characters of the previous one so meaning isn't cut in half."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_embedding_function():
    """Chroma's built-in embedding model (all-MiniLM-L6-v2, runs on your laptop, free).
    It downloads a small model (~80 MB) the first time."""
    from chromadb.utils import embedding_functions
    return embedding_functions.DefaultEmbeddingFunction()


# ---------------------------------------------------------------------------
# 1. O*NET
# ---------------------------------------------------------------------------
ONET_FILES = ["Task Statements.txt", "Task Ratings.txt"]


def download_onet():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((config.RAW_DIR / f).exists() for f in ONET_FILES):
        print("  O*NET files already present - skipping download.")
        return
    for v in config.ONET_VERSIONS:
        url = config.ONET_ZIP_URL.format(v=v)
        print(f"  Trying O*NET {v.replace('_', '.')} ...")
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
        except requests.RequestException as e:
            print(f"    could not connect ({e.__class__.__name__})")
            continue
        if r.status_code != 200:
            print(f"    not available (HTTP {r.status_code})")
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for wanted in ONET_FILES:
                member = next((n for n in z.namelist() if n.endswith("/" + wanted) or n == wanted), None)
                if member is None:
                    raise RuntimeError(f"{wanted} not found inside the O*NET zip")
                (config.RAW_DIR / wanted).write_bytes(z.read(member))
        print(f"  Downloaded O*NET {v.replace('_', '.')}.")
        return
    sys.exit(
        "\nCould not download O*NET automatically.\n"
        "Do this by hand:\n"
        "  1. Open https://www.onetcenter.org/database.html\n"
        "  2. Download the 'Text' zip of the latest database\n"
        f"  3. Copy 'Task Statements.txt' and 'Task Ratings.txt' into: {config.RAW_DIR}\n"
        "  4. Run  python data_prep.py  again.\n"
    )


def load_onet():
    statements = pd.read_csv(config.RAW_DIR / "Task Statements.txt", sep="\t", dtype=str)
    ratings = pd.read_csv(config.RAW_DIR / "Task Ratings.txt", sep="\t", dtype=str)
    # Importance (Scale ID "IM") is rated 1-5 by job holders
    imp = ratings[ratings["Scale ID"] == "IM"][["O*NET-SOC Code", "Task ID", "Data Value"]]
    imp = imp.rename(columns={"Data Value": "importance"})
    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce")
    df = statements[["O*NET-SOC Code", "Task ID", "Task", "Task Type"]].merge(
        imp, on=["O*NET-SOC Code", "Task ID"], how="left"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Anthropic Economic Index
# ---------------------------------------------------------------------------
def download_csv(url: str, target):
    if target.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            target.write_bytes(r.content)
            return True
        print(f"    HTTP {r.status_code} for {url}")
    except requests.RequestException as e:
        print(f"    could not connect ({e.__class__.__name__})")
    return False


def load_economic_index():
    """Returns a table: task_key, ai_usage_pct, automation_share, augmentation_share.
    If the download fails the app still works - it just has less evidence."""
    pct_file = config.RAW_DIR / "task_pct_v2.csv"
    aa_file = config.RAW_DIR / "automation_vs_augmentation_by_task.csv"
    ok1 = download_csv(config.ECON_TASK_PCT, pct_file)
    ok2 = download_csv(config.ECON_AUTO_AUG, aa_file)
    if not ok1:
        print("  WARNING: Economic Index not available - continuing without AI-usage data.")
        return None
    pct = pd.read_csv(pct_file)
    pct["pct"] = pd.to_numeric(pct["pct"], errors="coerce").fillna(0.0)
    pct["task_key"] = pct["task_name"].map(normalize_task)
    pct = pct.groupby("task_key", as_index=False)["pct"].sum().rename(columns={"pct": "ai_usage_pct"})
    if ok2:
        aa = pd.read_csv(aa_file)
        aa["task_key"] = aa["task_name"].map(normalize_task)
        for col in ["directive", "feedback_loop", "validation", "task_iteration", "learning"]:
            if col not in aa.columns:
                aa[col] = 0.0
            aa[col] = pd.to_numeric(aa[col], errors="coerce").fillna(0.0)
        # Anthropic's definition: directive + feedback loop = AUTOMATION;
        # validation + task iteration + learning = AUGMENTATION
        aa["automation_share"] = aa["directive"] + aa["feedback_loop"]
        aa["augmentation_share"] = aa["validation"] + aa["task_iteration"] + aa["learning"]
        for col in ["automation_share", "augmentation_share"]:
            aa[col] = pd.to_numeric(aa[col], errors="coerce").astype("float64")
        total = aa["automation_share"] + aa["augmentation_share"]
        total = total.where(total > 0)          # 0 -> missing (avoids dividing by zero)
        aa["automation_share"] = (aa["automation_share"] / total).round(2)
        aa["augmentation_share"] = (aa["augmentation_share"] / total).round(2)
        aa = aa.groupby("task_key", as_index=False)[["automation_share", "augmentation_share"]].mean()
        pct = pct.merge(aa, on="task_key", how="left")
    return pct


# ---------------------------------------------------------------------------
# 3. Build the tasks table for the 5 roles
# ---------------------------------------------------------------------------
def build_tasks_table():
    onet = load_onet()
    rows = []
    for role, info in config.ROLES.items():
        sub = onet[onet["O*NET-SOC Code"] == info["onet_code"]].copy()
        if sub.empty:
            print(f"  WARNING: no O*NET tasks found for {role} ({info['onet_code']}).")
            continue
        sub["role"] = role
        sub["onet_title"] = info["onet_title"]
        rated = sub["importance"].notna().sum()
        print(f"  {role}: {len(sub)} tasks, {rated} with importance ratings"
              + ("" if rated else "  <- no ratings: using Core tasks, equal weights"))
        rows.append(sub)
    tasks = pd.concat(rows, ignore_index=True)
    tasks = tasks.rename(columns={
        "O*NET-SOC Code": "onet_code", "Task ID": "task_id", "Task": "task", "Task Type": "task_type",
    })
    tasks["task_key"] = tasks["task"].map(normalize_task)

    econ = load_economic_index()
    if econ is not None:
        tasks = tasks.merge(econ, on="task_key", how="left")
        tasks["ai_usage_pct"] = tasks["ai_usage_pct"].fillna(0.0)
    else:
        tasks["ai_usage_pct"] = 0.0
    for col in ["automation_share", "augmentation_share"]:
        if col not in tasks.columns:
            tasks[col] = pd.NA

    tasks = tasks.sort_values(["role", "importance"], ascending=[True, False])
    cols = ["role", "onet_code", "onet_title", "task_id", "task", "task_type", "importance",
            "ai_usage_pct", "automation_share", "augmentation_share"]
    tasks[cols].to_csv(config.TASKS_CSV, index=False)
    print(f"  Saved {len(tasks)} tasks for {tasks['role'].nunique()} roles -> {config.TASKS_CSV.name}")
    matched = (tasks["ai_usage_pct"] > 0).sum()
    print(f"  {matched} of these tasks have AI-usage data from the Economic Index.")


# ---------------------------------------------------------------------------
# 4. Build the RAG index over ZenML case studies
# ---------------------------------------------------------------------------
def load_case_studies() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(config.ZENML_DATASET, split="train")
    return ds.to_pandas()


def build_case_index(embedding_function=None, cases: pd.DataFrame = None):
    import chromadb

    if cases is None:
        print("  Downloading ZenML case studies from Hugging Face ...")
        cases = load_case_studies()
    cases = cases.fillna("")
    cases = cases.reset_index(drop=True)
    if "case_id" not in cases.columns:
        cases["case_id"] = cases.index.astype(str)

    keep = ["case_id", "company", "title", "industry", "year", "short_summary", "webflow_url", "source_url"]
    for col in keep:
        if col not in cases.columns:
            cases[col] = ""
    if not config.CASES_CSV.exists() or "case_id" not in pd.read_csv(config.CASES_CSV, nrows=1).columns:
        cases[keep].to_csv(config.CASES_CSV, index=False)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection("case_studies")
    except Exception:
        pass
    collection = client.create_collection(
        name="case_studies",
        embedding_function=embedding_function or get_embedding_function(),
        metadata={"hnsw:space": "cosine"},   # cosine similarity, as in class
    )

    ids, docs, metas = [], [], []
    for _, c in cases.iterrows():
        # Text to search over: title + company + summary (full summary if available)
        body = c["full_summary"] if "full_summary" in cases.columns and c["full_summary"] else c["short_summary"]
        header = f"{c['company']} - {c['title']}. Industry: {c['industry']}. "
        for j, chunk in enumerate(chunk_text(header + str(body))):
            ids.append(f"{c['case_id']}-{j}")
            docs.append(chunk)
            metas.append({
                "case_id": str(c["case_id"]),
                "company": str(c["company"]),
                "title": str(c["title"]),
                "industry": str(c["industry"]),
                "year": str(c["year"]),
                "url": str(c["webflow_url"] or c["source_url"]),
            })

    print(f"  Embedding {len(docs)} chunks from {len(cases)} case studies (this is the slow part) ...")
    batch = 256
    for i in range(0, len(docs), batch):
        collection.add(ids=ids[i:i + batch], documents=docs[i:i + batch], metadatas=metas[i:i + batch])
        done = min(i + batch, len(docs))
        print(f"    {done}/{len(docs)} chunks indexed", end="\r")
    print(f"\n  Vector database ready -> {config.CHROMA_DIR}")


def ensure_case_index():
    """Used on Streamlit Cloud: if there is no vector database yet, build a small one from
    data/cases.csv (short summaries, ~2,000 chunks, 1-2 minutes). Locally, data_prep.py has
    already built the full index from the complete case texts, so this does nothing."""
    if (config.CHROMA_DIR / "chroma.sqlite3").exists() or not config.CASES_CSV.exists():
        return False
    cases = pd.read_csv(config.CASES_CSV, dtype=str).fillna("")
    cases["full_summary"] = ""            # cloud index uses the short summaries only
    build_case_index(cases=cases)
    return True


# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tasks":      # quick: rebuild only the tasks table
        print("\nRebuilding the tasks table only (case-study database is kept)")
        build_tasks_table()
        print("\nDone.\n")
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] O*NET task data")
    download_onet()
    print("\n[2/3] Tasks table for the 5 roles (+ Anthropic Economic Index)")
    build_tasks_table()
    print("\n[3/3] Case-study vector database (RAG)")
    build_case_index()
    print("\nAll done. Now start the app with:   streamlit run app.py\n")


if __name__ == "__main__":
    main()
