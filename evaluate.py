"""
evaluate.py - STEP 3 (for your report). Measures the success criteria.

  python evaluate.py templates            -> creates label sheets in labels/ for you to fill in
  python evaluate.py run [N] [model]      -> runs the agent on all 5 roles N times (default 1),
                                             e.g.  python evaluate.py run 1 gemini-3.5-flash-lite
                                             Already-finished runs are skipped, so you can re-run it
                                             after a quota stop and it continues where it left off.
  python evaluate.py score                -> agreement with your labels, consistency, evidence, speed

Runs you made in the app (saved in outputs/) are counted automatically.
The API key is read from .streamlit/secrets.toml (same as the app) or GEMINI_API_KEY.
"""
import os
import re
import sys

import pandas as pd

import config

VERDICTS = {"automate": "Automate", "augment": "Augment", "human-only": "Human-only",
            "human only": "Human-only", "humanonly": "Human-only", "human": "Human-only"}


def _norm(v):
    return VERDICTS.get(str(v).strip().lower(), str(v).strip())


def _api_key(provider):
    key_name = config.PROVIDERS[provider]["key_name"]
    key = os.environ.get(key_name)
    secrets = config.BASE_DIR / ".streamlit" / "secrets.toml"
    if not key and secrets.exists():          # same key file the app uses
        import tomllib
        key = tomllib.loads(secrets.read_text(encoding="utf-8-sig")).get(key_name)
    return key


def run_all(runs: int = 1, model: str = None):
    from agent import run_agent
    provider = config.DEFAULT_PROVIDER
    key = _api_key(provider)
    if not key:
        sys.exit("No API key found. Save it in .streamlit/secrets.toml (see README) or set GEMINI_API_KEY.")
    model = model or config.PROVIDERS[provider]["models"][0]
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "-", model.lower())
    for i in range(1, runs + 1):
        for role in config.ROLES:
            out = config.OUTPUT_DIR / f"eval_{role.replace(' ', '_')}_{tag}_run{i}.csv"
            if out.exists():
                print(f"Skip {role}, run {i} ({model}) - already done")
                continue
            print(f"Running {role}, run {i}/{runs} with {model} ...")
            try:
                res = run_agent(role, api_key=key, model=model, provider=provider)
            except RuntimeError as e:
                print(f"\nSTOPPED: {e}\nYour finished runs are saved. Re-run later or with another model, e.g.:\n"
                      f"   python evaluate.py run {runs} gemini-3.5-flash-lite\n")
                return
            t = res["table"]
            t["role"], t["seconds"], t["model"] = role, res["seconds"], model
            t.to_csv(out, index=False)
            print(f"   done in {res['seconds']} s, shares: {res['shares']}")


def make_templates():
    """One sheet per role with the same tasks the agent analyses. Fill 'your_verdict'
    with Automate / Augment / Human-only BEFORE looking at the agent's answers."""
    from tools import select_top_tasks
    config.LABELS_DIR.mkdir(exist_ok=True)
    tasks = pd.read_csv(config.TASKS_CSV, dtype={"task_id": str})
    for role in config.ROLES:
        sub = select_top_tasks(tasks[tasks["role"] == role])
        sheet = sub[["task_id", "task", "importance"]].copy()
        sheet["your_verdict"] = ""
        sheet["classmate_verdict"] = ""
        path = config.LABELS_DIR / f"labels_{role.replace(' ', '_')}.csv"
        if path.exists():
            print(f"Exists, not overwritten: {path.name}")
            continue
        sheet.to_csv(path, index=False)
        print(f"Created {path.name}")
    print("\nOpen these in Excel, fill the verdict columns, save as CSV (same name).")


def _load_runs() -> pd.DataFrame:
    """All agent results: evaluation runs + runs saved by the app."""
    frames = []
    roles = {r.replace(" ", "_"): r for r in config.ROLES}
    for f in sorted(config.OUTPUT_DIR.glob("*.csv")):
        if f.name in ("feedback.csv", "evaluation_summary.csv"):
            continue
        name = f.stem[5:] if f.stem.startswith("eval_") else f.stem
        role = next((r for k, r in roles.items() if name.startswith(k)), None)
        if role is None:
            continue
        d = pd.read_csv(f, dtype={"task_id": str})
        if "verdict" not in d.columns:
            continue
        d["role"], d["run"] = role, f.stem
        if "seconds" not in d.columns:
            d["seconds"] = float("nan")
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score():
    runs = _load_runs()
    if runs.empty:
        sys.exit("No agent results found in outputs/. Run some analyses in the app first.")
    runs["verdict"] = runs["verdict"].map(_norm)
    rows = []
    for role in config.ROLES:
        r = runs[runs["role"] == role]
        if r.empty:
            continue
        n_runs = r["run"].nunique()
        pivot = r.pivot_table(index="task_id", columns="run", values="verdict", aggfunc="first")
        consistent = (pivot.nunique(axis=1) == 1).mean() * 100 if n_runs > 1 else None
        evidence = (~r["evidence"].fillna("").str.lower().str.contains("no evidence")).mean() * 100
        case_link = (r["evidence_url"].fillna("").astype(str).str.len() > 0).mean() * 100

        # Agreement is scored PER RUN, then averaged (no majority vote, so no tie-breaking bias)
        you = mate = human = None
        lab_file = config.LABELS_DIR / f"labels_{role.replace(' ', '_')}.csv"
        if lab_file.exists():
            lab = pd.read_csv(lab_file, dtype={"task_id": str})
            for col in ["your_verdict", "classmate_verdict"]:
                lab[col] = lab[col].fillna("").map(_norm) if col in lab.columns else ""
            per_you, per_mate = [], []
            for _, run_df in r.groupby("run"):
                m = lab.merge(run_df[["task_id", "verdict"]], on="task_id", how="inner")
                y = m[m["your_verdict"] != ""]
                if not y.empty:
                    per_you.append((y["your_verdict"] == y["verdict"]).mean() * 100)
                c = m[m["classmate_verdict"] != ""]
                if not c.empty:
                    per_mate.append((c["classmate_verdict"] == c["verdict"]).mean() * 100)
            you = sum(per_you) / len(per_you) if per_you else None
            mate = sum(per_mate) / len(per_mate) if per_mate else None
            b = lab[(lab["your_verdict"] != "") & (lab["classmate_verdict"] != "")]
            if not b.empty:
                human = (b["your_verdict"] == b["classmate_verdict"]).mean() * 100
        rnd = lambda x: None if x is None else round(x, 1)
        rows.append({"role": role, "runs": n_runs,
                     "agent_vs_you_%": rnd(you), "agent_vs_classmate_%": rnd(mate), "you_vs_classmate_%": rnd(human),
                     "consistency_%": rnd(consistent), "evidence_cited_%": rnd(evidence),
                     "case_study_linked_%": rnd(case_link),
                     "avg_seconds": rnd(r.groupby("run")["seconds"].first().mean())})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(config.OUTPUT_DIR / "evaluation_summary.csv", index=False)

    # Overall + where you and the agent disagree (useful for the report)
    if out["agent_vs_you_%"].notna().any():
        print(f"\nOverall agreement with your labels: {out['agent_vs_you_%'].mean():.1f}%  (target >= 80%)")
    print(f"Saved {config.OUTPUT_DIR / 'evaluation_summary.csv'}")
    print("Targets: agreement >= 80%, consistency >= 90% (needs 2+ runs per role), evidence 100%, speed < 120 s.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        m = sys.argv[3] if len(sys.argv) > 3 else None
        run_all(n, m)
    elif cmd == "templates":
        make_templates()
    elif cmd == "score":
        score()
    else:
        print(__doc__)
