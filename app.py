"""
app.py - STEP 2. The Streamlit web app.

    streamlit run app.py

Opens in your browser. Pick a role, click Analyze, and watch the agent work.
"""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Job Task Impact Analyzer", page_icon="🔮", layout="wide")

VERDICT_COLORS = {"Automate": "#DDD6FE", "Augment": "#CCFBF1", "Human-only": "#FEF3C7"}
VERDICT_ICONS = {"Automate": "🤖", "Augment": "🤝", "Human-only": "🙋"}
ROLE_ICONS = {"HR Recruiter": "🧑‍💼", "Customer Support Agent": "🎧", "Credit Analyst": "💳",
              "Software QA Tester": "🧪", "Registered Nurse": "🩺"}

st.markdown("""
<style>
  .hero {background: linear-gradient(120deg, #7C3AED 0%, #2563EB 55%, #14B8A6 100%);
         border-radius: 20px; padding: 28px 32px; color: white; margin-bottom: 18px;}
  .hero h1 {color: white; font-size: 2.1rem; margin: 0 0 6px 0; padding: 0;}
  .hero p {color: #EEF2FF; font-size: 1.05rem; margin: 0;}
  .card {border-radius: 18px; padding: 18px 20px; text-align: center;
         box-shadow: 0 2px 10px rgba(0,0,0,0.06); color: #1F2937;}
  .card .emoji {font-size: 2rem;}
  .card .num {font-size: 2.2rem; font-weight: 800; margin: 2px 0;}
  .card .lbl {font-size: 0.95rem; font-weight: 600; opacity: 0.8;}
  .bar {display: flex; height: 26px; border-radius: 13px; overflow: hidden; margin: 14px 0 4px 0;}
  .bar div {display: flex; align-items: center; justify-content: center;
            font-size: 0.8rem; font-weight: 700; color: #1F2937;}
  .step {background: #F5F3FF; border-radius: 18px; padding: 18px; height: 100%; color: #1F2937;}
  .step .emoji {font-size: 1.8rem;}
  .step b {font-size: 1.05rem;}
  .pill {display: inline-block; background: #EDE9FE; color: #5B21B6; border-radius: 999px;
         padding: 4px 12px; font-size: 0.85rem; font-weight: 600; margin-right: 6px;}
</style>
""", unsafe_allow_html=True)


def get_api_key(key_name):
    """API key from Streamlit secrets (for deployment), environment, or the sidebar box."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.environ.get(key_name, "")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔮 Job Task Impact Analyzer")
    st.caption("How will AI change this job, task by task?")
    role_choice = st.selectbox("👤 Choose a role", list(config.ROLES.keys()),
                               format_func=lambda r: f"{ROLE_ICONS.get(r, '💼')}  {r}")
    custom = st.text_input("...or type a job title (must be one of the 5 roles)", "")
    jd = st.text_area("Optional: paste a job description for context", "", height=120)
    st.divider()
    provider = st.selectbox("🧠 LLM provider", list(config.PROVIDERS.keys()))
    pinfo = config.PROVIDERS[provider]
    key = get_api_key(pinfo["key_name"])
    if not key:
        key = st.text_input(f"{provider.split()[0]} API key", type="password", help=pinfo["key_help"])
    model = st.selectbox("⚙️ Model", pinfo["models"], index=0)
    go = st.button("✨ Analyze this job", type="primary", width="stretch")
    st.caption("Data: O*NET (US DoL, CC BY 4.0) · Anthropic Economic Index · ZenML LLMOps Database")

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
if not config.TASKS_CSV.exists() or not (config.CHROMA_DIR.exists() or config.CASES_CSV.exists()):
    st.error("Data not prepared yet. In the terminal run:  python data_prep.py  then restart the app.")
    st.stop()
if not (config.DATA_DIR / "chroma_ready.flag").exists():
    with st.spinner("🛠️ First start: building the case-study search index (about 1-2 minutes)..."):
        from data_prep import ensure_case_index
        ensure_case_index()

st.markdown("""
<div class="hero">
  <h1>🔮 Job Task Impact Analyzer</h1>
  <p>Pick a job. Our AI agent breaks it into real tasks and tells you which ones AI will
  <b>🤖 automate</b>, which it will <b>🤝 augment</b>, and which stay <b>🙋 human</b> &mdash;
  with real company evidence for every call.</p>
</div>
""", unsafe_allow_html=True)

if go:
    if not key:
        st.warning("Please enter your API key in the sidebar.")
        st.stop()
    job = custom.strip() or role_choice
    if jd.strip():
        # Pasted JD is passed as data, clearly marked (prompt-injection safety)
        job = f"{job}\n\n<job_description>\n{jd.strip()[:4000]}\n</job_description>"

    from agent import run_agent
    with st.status("🕵️ The agent is investigating this job...", expanded=True) as status:
        def on_step(text):
            st.write(text[:300])
        try:
            result = run_agent(job, api_key=key, model=model, provider=provider, on_step=on_step)
            status.update(label=f"✅ Done in {result['seconds']} s", state="complete", expanded=False)
            st.toast("Your job analysis is ready! 🎉")
        except Exception as e:
            status.update(label="😕 Something went wrong", state="error")
            st.error(f"{e.__class__.__name__}: {e}")
            st.stop()
    st.session_state["result"] = result

    # Save every run for evaluation (outputs/<role>_<time>.csv)
    if not result["table"].empty:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        role_name = (result["occupation"] or {}).get("role", "custom").replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result["table"].to_csv(config.OUTPUT_DIR / f"{role_name}_{stamp}.csv", index=False)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("result")
if result is None:
    st.markdown("### How it works")
    a, b, c = st.columns(3)
    a.markdown('<div class="step"><div class="emoji">👤</div><b>1. Pick a role</b><br>'
               'Choose one of 5 jobs in the sidebar and hit <b>Analyze</b>.</div>', unsafe_allow_html=True)
    b.markdown('<div class="step"><div class="emoji">🕵️</div><b>2. The agent investigates</b><br>'
               'It pulls official O*NET tasks, real AI-usage data and 2,000+ company case studies.</div>',
               unsafe_allow_html=True)
    c.markdown('<div class="step"><div class="emoji">🔮</div><b>3. See the job of the future</b><br>'
               'Every task gets a verdict, a reason and evidence &mdash; plus skills to build.</div>',
               unsafe_allow_html=True)
    st.caption("⏱️ One analysis takes about 30 seconds to 2 minutes on the free Gemini tier.")
    st.stop()

occ = result["occupation"]
if not occ or result["table"].empty:
    st.warning("This job is outside the project scope. Try one of: " + ", ".join(config.ROLES.keys()))
    st.stop()

icon = ROLE_ICONS.get(occ["role"], "💼")
st.markdown(f"## {icon} {occ['role']}")
st.markdown(f'<span class="pill">O*NET {occ["onet_code"]}</span><span class="pill">{occ["onet_title"]}</span>'
            f'<span class="pill">⏱️ {result["seconds"]} s</span>', unsafe_allow_html=True)
s = result["shares"]
cards = st.columns(3)
for col, (name, colour) in zip(cards, VERDICT_COLORS.items()):
    col.markdown(f'<div class="card" style="background:{colour}"><div class="emoji">{VERDICT_ICONS[name]}</div>'
                 f'<div class="num">{s.get(name, 0)}%</div><div class="lbl">{name}</div></div>',
                 unsafe_allow_html=True)
segments = "".join(
    f'<div style="width:{s.get(n, 0)}%;background:{c}">{round(s.get(n, 0))}%</div>'
    for n, c in VERDICT_COLORS.items() if s.get(n, 0) > 0)
st.markdown(f'<div class="bar">{segments}</div>', unsafe_allow_html=True)
st.caption(f"How this job's top {config.TOP_N_TASKS} tasks split, weighted by O*NET importance (1-5).")
st.markdown("### 📋 Task by task")

table = result["table"].copy()
show = table[["task", "importance", "verdict", "why", "evidence", "confidence",
              "routine", "information", "judgement", "human_touch", "evidence_url"]]
show["verdict"] = show["verdict"].map(lambda v: f"{VERDICT_ICONS.get(v, '')} {v}")
show = show.rename(columns={"task": "Task", "importance": "Importance", "verdict": "Verdict",
                            "why": "Why", "evidence": "Evidence", "confidence": "Confidence",
                            "routine": "Routine", "information": "Information",
                            "judgement": "Judgement", "human_touch": "Human touch",
                            "evidence_url": "Source link"})


def color_verdict(v):
    name = str(v).split(" ", 1)[-1]
    return f"background-color: {VERDICT_COLORS.get(name, '')}; color: #111; font-weight: 600"


st.dataframe(
    show.style.map(color_verdict, subset=["Verdict"]),
    width="stretch", hide_index=True,
    column_config={"Source link": st.column_config.LinkColumn("Source link"),
                   "Importance": st.column_config.NumberColumn(format="%.2f")},
)

left, right = st.columns(2)
with left:
    st.markdown("#### 🔄 How this role shifts")
    st.write(result["summary"])
with right:
    st.markdown("#### 🚀 Skills to build")
    for skill in result["skills"]:
        st.write(f"- {skill}")

with st.expander("🧠 How the agent thought (ReAct trace)"):
    for step in result["trace"]:
        st.markdown(f"**{step['step']}. {step['kind']}** — {step['text'][:800]}")

with st.expander("📖 Full O*NET task wording"):
    st.dataframe(table[["onet_task", "importance", "verdict"]], width="stretch", hide_index=True)

st.download_button("⬇️ Download results (CSV)", table.to_csv(index=False).encode("utf-8"),
                   file_name=f"{occ['role'].replace(' ', '_')}_analysis.csv", mime="text/csv")

st.markdown("#### 💬 Was this analysis useful?")
fb = st.feedback("thumbs")
if fb is not None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    fb_file = config.OUTPUT_DIR / "feedback.csv"
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"),
                         "role": occ["role"], "thumbs_up": bool(fb)}])
    row.to_csv(fb_file, mode="a", header=not fb_file.exists(), index=False)
    st.caption("Thanks — feedback saved! 🙏")

st.caption(f"Model: {result['model']} · tokens in/out: {result['tokens']['input']:,}/{result['tokens']['output']:,} · "
           "Decision support only: verdicts describe what AI can do with today's tools, not what every "
           "company will adopt. Case-study results are vendor-reported.")
