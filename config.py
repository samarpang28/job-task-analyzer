"""
config.py - All the settings for the Job Task Impact Analyzer in one place.

You normally do NOT need to change anything here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # prepared data lives here
RAW_DIR = DATA_DIR / "raw"              # downloaded original files
CHROMA_DIR = DATA_DIR / "chroma"        # the vector database (RAG)
OUTPUT_DIR = BASE_DIR / "outputs"       # agent results (CSV files)
LABELS_DIR = BASE_DIR / "labels"        # your human labels for evaluation

TASKS_CSV = DATA_DIR / "tasks.csv"      # the 5 roles' tasks, importance, AI usage
CASES_CSV = DATA_DIR / "cases.csv"      # list of case studies (for display)

# ---------------------------------------------------------------------------
# The 5 roles in scope, mapped to their official O*NET occupation codes
# ---------------------------------------------------------------------------
ROLES = {
    "HR Recruiter": {
        "onet_code": "13-1071.00",
        "onet_title": "Human Resources Specialists",
        "aliases": ["recruiter", "hr recruiter", "talent acquisition", "hr specialist", "hiring"],
    },
    "Customer Support Agent": {
        "onet_code": "43-4051.00",
        "onet_title": "Customer Service Representatives",
        "aliases": ["customer support", "customer service", "support agent", "call center", "call centre"],
    },
    "Credit Analyst": {
        "onet_code": "13-2041.00",
        "onet_title": "Credit Analysts",
        "aliases": ["credit analyst", "credit risk", "loan analyst", "credit underwriter",
                    "financial analyst", "banking analyst"],
    },
    "Software QA Tester": {
        "onet_code": "15-1253.00",
        "onet_title": "Software Quality Assurance Analysts and Testers",
        "aliases": ["qa", "quality assurance", "software tester", "test engineer", "qa developer", "tester"],
    },
    "Registered Nurse": {
        "onet_code": "29-1141.00",
        "onet_title": "Registered Nurses",
        "aliases": ["nurse", "registered nurse", "staff nurse", "nursing"],
    },
}

# How many of each role's most important tasks the agent analyses.
# (Keeps each run fast and cheap. State this as an assumption in your report.)
TOP_N_TASKS = 10

# ---------------------------------------------------------------------------
# Data download locations
# ---------------------------------------------------------------------------
# O*NET publishes a zip of text files per version. We try the newest first.
ONET_VERSIONS = ["31_0", "30_2", "30_1", "30_0", "29_3", "29_2", "29_1", "29_0"]
ONET_ZIP_URL = "https://www.onetcenter.org/dl_files/database/db_{v}_text.zip"

# Anthropic Economic Index (release of 27 March 2025): per-task AI usage
ECON_BASE = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/release_2025_03_27/"
ECON_TASK_PCT = ECON_BASE + "task_pct_v2.csv"
ECON_AUTO_AUG = ECON_BASE + "automation_vs_augmentation_by_task.csv"

# ZenML LLMOps Database on Hugging Face (2,000+ real company case studies)
ZENML_DATASET = "zenml/llmops-database"

# ---------------------------------------------------------------------------
# RAG settings (Sessions 7-8: chunk size, overlap, top-k)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 100     # characters shared between neighbouring chunks
TOP_K = 4               # how many case-study chunks to return per search
RETRIEVAL_MODE = "hybrid"   # "vector" (embeddings only), "keyword" (BM25 only) or "hybrid" (both, fused)

# ---------------------------------------------------------------------------
# LLM providers (pick one in the app's sidebar)
# Gemini (Google AI Studio) has a FREE tier - the default for this project.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Gemini (free)": {
        "key_name": "GEMINI_API_KEY",
        "key_help": "Free key from https://aistudio.google.com -> Get API key",
        "models": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
    },
    "Claude (paid)": {
        "key_name": "ANTHROPIC_API_KEY",
        "key_help": "Key from https://console.anthropic.com (needs credit)",
        "models": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
    },
}
DEFAULT_PROVIDER = "Gemini (free)"
MAX_AGENT_TURNS = 30        # safety limit on the ReAct loop
GEMINI_MIN_SECONDS = 6.5    # free tier allows ~10 requests/minute, so we pace calls
