# Job Task Impact Analyzer — How to run it

An AI agent (Gemini or Claude + 5 tools + RAG) that breaks a job into its O*NET tasks and judges which
tasks AI will **automate**, **augment** or leave **human-only**, with real company evidence.

## What each file does

| File | What it is |
| --- | --- |
| `config.py` | Settings: the 5 roles, RAG chunk size/overlap/top-k, model names |
| `data_prep.py` | **Run once.** Downloads O*NET + Economic Index + ZenML cases, builds the vector DB |
| `tools.py` | The agent's 5 tools (incl. the RAG search) and the fixed decision rule |
| `retrieval.py` | Hybrid RAG: vector search + BM25 keyword search, fused with Reciprocal Rank Fusion |
| `retrieval_eval.py` | Before/after test: vector-only vs hybrid retrieval, blind-judged by Gemini (5 requests) |
| `make_deploy_folder.py` | Collects the files needed for Streamlit Cloud (see DEPLOY.md) |
| `agent.py` | The ReAct loop: the LLM thinks → calls a tool → reads the result → repeats (Gemini or Claude) |
| `app.py` | The Streamlit web app |
| `evaluate.py` | Measures the success criteria for your report |

---

## Step 0 — One-time setup (15 min)

1. Install **Python 3.13** (Windows installer 64-bit) from https://www.python.org/downloads/windows/
   - Windows: on the first installer screen, tick **"Add python.exe to PATH"**.
2. Get a **free Gemini API key** (no credit card):
   - Go to https://aistudio.google.com and sign in with your Google account.
   - Click **Get API key** → **Create API key** → copy it (it starts with `AIza`).
   - Keep it private, like a password.
3. Unzip this folder somewhere easy, e.g. `Documents\job-task-analyzer`.
4. Open a terminal **in that folder**:
   - Windows: open the folder in File Explorer, click the address bar, type `cmd`, press Enter.
   - Mac: right-click the folder → *New Terminal at Folder*.

## Step 1 — Install the packages (5–10 min)

Copy-paste these lines one at a time.

**Windows**
```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --only-binary=:all:
```

**Mac**
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt --only-binary=:all:
```

You'll see `(venv)` at the start of the line. Each time you open a new terminal, run the
`activate` line again before anything else.

## Step 2 — Prepare the data (10–15 min, once)

```
python data_prep.py
```

It prints `[1/3] … [2/3] … [3/3] … All done.` If O*NET fails to download, it tells you exactly
which two files to download by hand and where to put them.

## Step 3 — Start the app

```
streamlit run app.py
```

Your browser opens. In the sidebar keep **LLM provider = Gemini (free)**, paste your key,
pick a role, click **Analyze**. One analysis takes about 1–3 minutes, because the free tier
allows ~10 requests per minute and the app paces itself.
Stop the app with `Ctrl + C` in the terminal.

## Step 4 — Evaluation numbers for the report

The key is read from `.streamlit/secrets.toml` (same file as the app).

```
python evaluate.py templates                     # creates labels/*.csv - fill 'your_verdict' in Excel FIRST
python evaluate.py score                         # agreement, consistency, evidence, speed
python evaluate.py run 1 gemini-3.5-flash-lite   # optional: a 2nd run of all 5 roles, for consistency
python evaluate.py score
```
Runs you made in the app are counted automatically. The free tier allows about 20 requests per
model per day (one analysis = 4 requests); if a quota runs out, re-run the same command later or
with another model - finished runs are skipped.

## Step 4b — Hybrid RAG before/after test

```
python retrieval_eval.py
```
Compares vector-only and hybrid retrieval on all 50 tasks (blind LLM judge, 5 requests).

## Step 5 — Put it online (optional, for the demo link)

See **DEPLOY.md** for the step-by-step guide. Short version:

1. Create a free GitHub account and a new repository; upload this folder **including the
   `data` folder** but **not** `venv`.
2. Go to https://share.streamlit.io → *New app* → pick the repository → main file `app.py`.
3. In *Advanced settings → Secrets* add:  `GEMINI_API_KEY = "AIza..."`
4. Deploy. You get a public link.

Never put your API key inside the code or upload it to GitHub.

## If something goes wrong

| Message | Fix |
| --- | --- |
| `'python' is not recognized` | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python` |
| `Data not prepared yet` in the app | Run `python data_prep.py` first |
| `API key not valid` / `PERMISSION_DENIED` | Copy the key again from AI Studio |
| `429` / `RESOURCE_EXHAUSTED` | Free-tier limit hit: wait a minute (the app retries automatically), or pick `gemini-3.5-flash-lite` |
| `404` / model not found | Pick a different model in the sidebar |
| `Failed to build 'jiter'` / `maturin` | Use the `--only-binary=:all:` install line above; the Claude package is optional |
| Anything else | Copy the full error text and send it to Claude |

## Data credits

- O*NET Database, U.S. Department of Labor, CC BY 4.0
- Anthropic Economic Index (release 2025-03-27), CC-BY
- ZenML LLMOps Database (Hugging Face), Apache-2.0

Note: Google may use free-tier Gemini inputs to improve its models. This project only sends
public data (O*NET tasks, published case studies), so that is acceptable here.
