# Streamlit chat UI + RAG + Precomputed Patient Summary Cards
#
# Run:
#   streamlit run app_chat.py

from __future__ import annotations

import json
import os
import pickle
import re
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer


# ====================== Fixed demo configuration ======================
APP_TITLE = "Neonatal ECMO RAG Chat + Early Warning"
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./store_txt_rag")).resolve()
PRECOMPUTED_CARDS = Path(os.getenv("PRECOMPUTED_CARDS", "./precomputed_cards.json"))
SAMPLE_CSV = Path(os.getenv("SAMPLE_CSV", "./sim_ecmo_timeseries.csv"))

STEP_SEC = 15
RAG_TOPK = int(os.getenv("RAG_TOPK", "6"))
EWMA_ALPHA = float(os.getenv("EWMA_ALPHA", "0.35"))
ALARM_THRESHOLD = float(os.getenv("ALARM_THRESHOLD", "0.50"))
HOLD_MIN = float(os.getenv("HOLD_MIN", "1.0"))
REFRACTORY_MIN = float(os.getenv("REFRACTORY_MIN", "10.0"))

DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")
USE_OLLAMA = os.getenv("USE_OLLAMA", "1") == "1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


# ====================== Utility ======================
def get_secret(name: str, default: str = "") -> str:
    """Read Streamlit secrets first, then environment variables."""
    try:
        val = st.secrets.get(name, None)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, default)


def _np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.index, pd.DatetimeIndex):
        return d.sort_index()
    if "AR" in d.columns:
        d["AR"] = pd.to_datetime(d["AR"], errors="coerce")
        d = d.dropna(subset=["AR"]).set_index("AR").sort_index()
    return d


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def extract_between(text: str, start: str, end: str) -> str:
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    m = re.search(pattern, text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


# ====================== RAG store ======================
class RagStore:
    def __init__(self, store_dir: Path):
        self.dir = store_dir
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X: Optional[sp.csr_matrix] = None

    def load(self) -> "RagStore":
        texts_fp = self.dir / "texts.jsonl"
        metas_fp = self.dir / "metas.jsonl"
        vect_fp = self.dir / "vectorizer.pkl"
        xtfidf_fp = self.dir / "X_tfidf.npz"
        missing = [p.name for p in [texts_fp, metas_fp, vect_fp, xtfidf_fp] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing RAG files in {self.dir}: {', '.join(missing)}")

        with texts_fp.open("r", encoding="utf-8") as f:
            self.texts = []
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.texts.append(obj.get("text", "") if isinstance(obj, dict) else str(obj))

        with metas_fp.open("r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
        self.metas = [r.get("meta", r) if isinstance(r, dict) else {"source": str(r)} for r in raw]

        with vect_fp.open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.X = sp.load_npz(xtfidf_fp).tocsr()

        n = min(len(self.texts), len(self.metas), self.X.shape[0])
        self.texts = self.texts[:n]
        self.metas = self.metas[:n]
        self.X = self.X[:n]
        return self

    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo + 1e-9)

    def search(self, query: str, topk: int = RAG_TOPK) -> List[Dict[str, Any]]:
        if self.vectorizer is None or self.X is None or not self.texts:
            return []
        qv = self.vectorizer.transform([query])
        score = (self.X @ qv.T).toarray().ravel()
        score = self._minmax01(score)
        order = np.argsort(-score)[:topk]
        rows: List[Dict[str, Any]] = []
        for rank, idx in enumerate(order, 1):
            rows.append(
                {
                    "rank": rank,
                    "score": float(score[idx]),
                    "text": self.texts[idx],
                    "meta": self.metas[idx],
                }
            )
        return rows


@st.cache_resource(show_spinner=False)
def load_rag_store(path: str) -> Optional[RagStore]:
    try:
        return RagStore(Path(path)).load()
    except Exception:
        return None


# ====================== Risk and alarm ======================
def proxy_risk(df: pd.DataFrame) -> np.ndarray:
    """Lightweight risk score for web demo uploads."""
    n = len(df)
    terms: List[np.ndarray] = []

    def ffill_col(name: str, limit: int = 240) -> Optional[pd.Series]:
        if name not in df.columns:
            return None
        return safe_numeric(df[name]).ffill(limit=limit)

    o2 = ffill_col("O2_sat")
    if o2 is not None:
        terms.append(0.45 * _sig((72 - o2.to_numpy()) / 4.0))

    sbp = ffill_col("SBP")
    if sbp is not None:
        terms.append(0.35 * _sig((58 - sbp.to_numpy()) / 4.0))

    dbp = ffill_col("DBP")
    if dbp is not None:
        terms.append(0.20 * _sig((32 - dbp.to_numpy()) / 3.0))

    hr = ffill_col("HR")
    if hr is not None:
        hrv = hr.to_numpy()
        terms.append(0.15 * np.maximum(_sig((hrv - 190) / 6.0), _sig((60 - hrv) / 6.0)))

    nirs = ffill_col("NIRS")
    if nirs is not None:
        terms.append(0.20 * _sig((58 - nirs.to_numpy()) / 5.0))

    lactate = ffill_col("Lactate")
    if lactate is not None:
        terms.append(0.25 * _sig((lactate.to_numpy() - 4.0) / 1.0))

    if not terms:
        return np.zeros(n, dtype=float)

    X = np.vstack([np.nan_to_num(t, nan=0.0) for t in terms])
    risk = 1.0 - np.prod(1.0 - X, axis=0)
    return np.clip(risk, 0.0, 1.0)


def ewma(x: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    x = _np(x).ravel()
    y = np.zeros_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x):
        acc = alpha * float(v) + (1.0 - alpha) * acc
        y[i] = acc
    return y


def alarm_indices(smooth: np.ndarray, threshold: float = ALARM_THRESHOLD) -> List[int]:
    hold_steps = max(1, int(round(HOLD_MIN * 60 / STEP_SEC)))
    ref_steps = max(1, int(round(REFRACTORY_MIN * 60 / STEP_SEC)))
    alarms: List[int] = []
    consec = 0
    i = 0
    while i < len(smooth):
        if smooth[i] >= threshold:
            consec += 1
            if consec >= hold_steps:
                alarms.append(i)
                i += ref_steps
                consec = 0
                continue
        else:
            consec = 0
        i += 1
    return alarms


def six_hour_risk_summary(smooth: np.ndarray) -> float:
    s = _np(smooth).ravel()
    if s.size < 8:
        return float(np.clip(s[-1] if s.size else 0.02, 0.001, 0.99))
    r_now = float(np.clip(s[-1], 0.0, 1.0))
    w1h = max(2, int(round(3600 / STEP_SEC)))
    seg = s[-w1h:] if s.size >= w1h else s
    r_mean = float(np.nanmean(seg))
    r_max = float(np.nanmax(seg))
    k = max(2, int(round(1800 / STEP_SEC)))
    sub = s[-k:] if s.size >= k else s
    x = np.arange(len(sub), dtype=float)
    x -= x.mean()
    denom = float((x * x).sum()) or 1.0
    slope = float((x * (sub - sub.mean())).sum() / denom)
    z = -2.0 + 2.2 * r_now + 1.4 * r_mean + 1.0 * r_max + 0.8 * max(0.0, slope)
    return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.001, 0.99))


def make_patient_summary(df: pd.DataFrame, smooth: Optional[np.ndarray] = None, p6: Optional[float] = None) -> str:
    d = ensure_datetime_index(df)
    tail = d.tail(max(1, int(round(30 * 60 / STEP_SEC))))
    cols = [c for c in ["HR", "RR", "O2_sat", "NIRS", "SBP", "DBP", "pH", "Lactate", "PAO2", "BE"] if c in tail.columns]
    stats: Dict[str, Dict[str, float]] = {}
    for c in cols:
        s = safe_numeric(tail[c])
        if s.notna().sum() == 0:
            continue
        stats[c] = {
            "last": round(float(s.dropna().iloc[-1]), 2),
            "mean": round(float(s.mean()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
        }

    flags = []
    if "O2_sat" in stats and stats["O2_sat"]["last"] < 72:
        flags.append("low oxygen saturation")
    if "SBP" in stats and stats["SBP"]["last"] < 58:
        flags.append("low systolic blood pressure")
    if "NIRS" in stats and stats["NIRS"]["last"] < 58:
        flags.append("low NIRS")
    if "Lactate" in stats and stats["Lactate"]["last"] > 4:
        flags.append("elevated lactate")

    lines = [
        "Recent patient summary from uploaded time-series:",
        "- Time window summarized: last ~30 minutes",
        f"- Current risk index: {float(smooth[-1]):.3f}" if smooth is not None and len(smooth) else "- Current risk index: unavailable",
        f"- Six-hour risk summary: {p6:.1%}" if p6 is not None else "- Six-hour risk summary: unavailable",
        f"- Active flags: {', '.join(flags) if flags else 'none detected by demo thresholds'}",
        f"- Vitals/labs statistics: {json.dumps(stats, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


# ====================== LLM generation ======================
def offline_template_answer(prompt: str) -> str:
    patient_summary = extract_between(prompt, "Patient summary:", "Retrieved ECMO evidence snippets:")
    evidence = extract_between(prompt, "Retrieved ECMO evidence snippets:", "Please answer")
    if not patient_summary:
        patient_summary = "No patient summary is currently available. Load a sample or upload a CSV and run analysis first."
    evidence_line = "RAG evidence was retrieved and used as local context." if evidence and "No RAG evidence" not in evidence else "No matching RAG evidence was retrieved for this question."
    return (
        "**LLM offline.** Based on the available patient summary and local RAG evidence, "
        "here is a template-based monitoring summary.\n\n"
        f"{patient_summary}\n\n"
        f"- Evidence context: {evidence_line}\n"
        "- Focus review on oxygenation, blood pressure, heart rate, cerebral NIRS, lactate, and the recent trend of the smoothed risk trajectory.\n"
        "- If the risk trajectory rises or multiple red flags appear together, consider closer monitoring, repeat assessment of recent labs, and readiness for escalation.\n"
        "- This is a demonstration workflow and not validated clinical advice."
    )


def call_ollama(prompt: str, timeout: int = 90) -> Optional[str]:
    if not USE_OLLAMA:
        return None
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return (resp.json().get("response") or "").strip()
    except Exception:
        return None
    return None


def call_huggingface(prompt: str, timeout: int = 120) -> str:
    token = get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")
    model = get_secret("HF_MODEL", DEFAULT_HF_MODEL)
    provider = get_secret("HF_PROVIDER", DEFAULT_HF_PROVIDER) or "auto"

    if not token:
        return offline_template_answer(prompt)

    try:
        from huggingface_hub import InferenceClient
    except Exception:
        return offline_template_answer(prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful clinical decision-support demo assistant. "
                "Use the patient summary and retrieved ECMO evidence. "
                "Do not diagnose. Give concise monitoring-oriented suggestions. "
                "Always mention this is a demo workflow, not validated clinical advice."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        try:
            client = InferenceClient(provider=provider, api_key=token, timeout=timeout)
        except TypeError:
            client = InferenceClient(model=model, token=token, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=550,
            temperature=0.25,
            top_p=0.9,
        )
        content = resp.choices[0].message.content
        if content:
            return str(content).strip()
    except Exception:
        pass

    try:
        try:
            client = InferenceClient(provider=provider, api_key=token, timeout=timeout)
            out = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=550,
                temperature=0.25,
                top_p=0.9,
            )
        except TypeError:
            client = InferenceClient(model=model, token=token, timeout=timeout)
            out = client.chat_completion(
                messages=messages,
                max_tokens=550,
                temperature=0.25,
                top_p=0.9,
            )
        content = out.choices[0].message.content
        if content:
            return str(content).strip()
    except Exception:
        pass

    return offline_template_answer(prompt)


def generate_answer(prompt: str) -> str:
    local = call_ollama(prompt)
    if local:
        return local
    return call_huggingface(prompt)


def build_prompt(user_question: str, patient_summary: str, snippets: List[Dict[str, Any]]) -> str:
    evidence_blocks = []
    for h in snippets:
        m = h.get("meta", {}) or {}
        src = m.get("source") or m.get("file_name") or "local corpus"
        page = m.get("page", "")
        text = (h.get("text") or "").replace("\n", " ").strip()
        evidence_blocks.append(f"[{h.get('rank', '?')}] {src}, page {page}: {text[:900]}")

    evidence = "\n\n".join(evidence_blocks) if evidence_blocks else "No RAG evidence retrieved."
    return f"""
User question:
{user_question}

Patient summary:
{patient_summary}

Retrieved ECMO evidence snippets:
{evidence}

Please answer in 1 short paragraph plus 3-5 bullets. Be practical and cautious. Mention when this is a demo/synthetic or uploaded data workflow rather than validated clinical advice.
""".strip()


# ====================== Cards and visualization ======================
@st.cache_data(show_spinner=False)
def load_cards(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [c for c in data if isinstance(c, dict) and c.get("question") and c.get("answer")]
    except Exception:
        return []


def risk_plot(df: pd.DataFrame, risk: np.ndarray, smooth: np.ndarray, alarms: List[int]) -> go.Figure:
    d = ensure_datetime_index(df)
    x = d.index if isinstance(d.index, pd.DatetimeIndex) else np.arange(len(d))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=risk, mode="lines", name="Raw risk", line=dict(width=1.2, color="#94a3b8")))
    fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", name="Smoothed risk", line=dict(width=3, color="#0f766e")))
    fig.add_hline(y=ALARM_THRESHOLD, line_dash="dot", line_color="#dc2626", annotation_text="Alarm threshold", annotation_position="top right")
    if alarms:
        ai = [i for i in alarms if i < len(smooth)]
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in ai],
                y=[smooth[i] for i in ai],
                mode="markers",
                name="Alarms",
                marker=dict(size=10, symbol="diamond", color="#dc2626"),
            )
        )
    ymax = max(0.2, min(1.0, float(np.nanmax(smooth)) * 1.35 if len(smooth) else 0.2))
    fig.update_layout(
        height=360,
        margin=dict(l=12, r=12, t=18, b=12),
        yaxis=dict(range=[0, ymax], title="Risk index"),
        xaxis=dict(title="Time"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.25)")
    return fig


# ====================== UI helpers ======================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f172a;
            --muted: #526174;
            --line: #c8d3df;
            --bg: #e7edf5;
            --panel: #f8fafc;
            --panel-2: #eef4fa;
            --navy: #0b1f35;
            --navy-2: #12304d;
            --teal: #0f766e;
            --accent: #1d4ed8;
        }

        /* Keep Streamlit's sidebar collapse / reopen control visible. */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[kind="header"] {
            visibility: visible !important;
            opacity: 1 !important;
            display: flex !important;
            z-index: 999999 !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 6%, rgba(15,118,110,.12), transparent 30%),
                radial-gradient(circle at 88% 4%, rgba(29,78,216,.10), transparent 28%),
                linear-gradient(180deg, #dfe8f3 0%, var(--bg) 38%, #d8e2ec 100%);
        }
        .block-container {
            padding-top: 2.75rem !important;
            padding-bottom: 3rem;
            max-width: 1340px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1f35 0%, #102a43 52%, #0f263d 100%);
            border-right: 1px solid rgba(255,255,255,.10);
        }
        [data-testid="stSidebar"] * {
            color: rgba(248,250,252,.92) !important;
        }
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: rgba(226,232,240,.72) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(226,232,240,.18) !important;
        }
        [data-testid="stSidebar"] .stFileUploader section {
            background: rgba(255,255,255,.07) !important;
            border: 1px dashed rgba(203,213,225,.45) !important;
            border-radius: 8px !important;
        }

        /* File uploader: make Upload text/icon visible and remove the odd inner square. */
        [data-testid="stSidebar"] .stFileUploader button {
            background: #1b3954 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(203,213,225,.45) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] .stFileUploader button * {
            background: transparent !important;
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }
        [data-testid="stSidebar"] .stFileUploader button svg,
        [data-testid="stSidebar"] .stFileUploader button svg * {
            background: transparent !important;
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }
        [data-testid="stSidebar"] .stFileUploader [data-testid="stIconMaterial"],
        [data-testid="stSidebar"] .stFileUploader [data-testid="stIconMaterial"] * {
            background: transparent !important;
        }
        [data-testid="stSidebar"] .stFileUploader button:hover,
        [data-testid="stSidebar"] .stFileUploader button:hover * {
            background: #24506f !important;
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stSidebar"] textarea {
            background: rgba(255,255,255,.08) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(203,213,225,.30) !important;
        }
        [data-testid="stSidebar"] textarea::placeholder {
            color: rgba(226,232,240,.60) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .075) !important;
            padding: 14px !important;
        }

        .hero-panel {
            background: linear-gradient(135deg, #0b1f35 0%, #102a43 48%, #0f766e 130%);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 14px;
            padding: 30px 34px;
            box-shadow: 0 18px 46px rgba(15,23,42,.20);
            margin-bottom: 18px;
        }
        .eyebrow {
            color: #67e8f9;
            font-weight: 850;
            letter-spacing: .17em;
            text-transform: uppercase;
            font-size: .78rem;
            margin-bottom: .75rem;
        }
        .title {
            color: #f8fafc;
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: -.04em;
            margin: 0 0 .9rem 0;
        }
        .subtitle {
            color: rgba(226,232,240,.86);
            font-size: 1.03rem;
            line-height: 1.58;
            max-width: 880px;
        }
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--ink);
            font-weight: 850;
            font-size: 1.08rem;
            margin: 1.05rem 0 .65rem 0;
        }
        .section-header::before {
            content: "";
            width: 5px;
            height: 22px;
            background: var(--teal);
            border-radius: 3px;
            display: inline-block;
        }
        .section-kicker {
            color: #24435f;
            font-size: .80rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .14em;
            margin: 1.05rem 0 .65rem 0;
            padding-bottom: .38rem;
            border-bottom: 1px solid rgba(36,67,95,.22);
        }
        .info-panel {
            background: rgba(248,250,252,.92);
            border: 1px solid var(--line);
            border-left: 5px solid var(--teal);
            border-radius: 10px;
            padding: 12px 14px;
            color: var(--ink);
            font-size: .93rem;
            line-height: 1.45;
            margin-top: .55rem;
        }
        .primary-note {
            background:#f1f7fb;
            border:1px solid #bfd0dc;
            color:#24435f;
            border-left: 5px solid var(--teal);
            border-radius:10px;
            padding:12px 14px;
            font-size:.92rem;
            line-height:1.45;
            margin-bottom: 12px;
        }
        .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, .075);
            margin-bottom: 16px;
        }
        .card-title {
            font-weight: 850;
            font-size: 1.02rem;
            color: var(--ink);
            margin-bottom: 6px;
        }
        .card-subtitle {
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.38;
            margin-bottom: 10px;
        }
        div.stButton > button {
            border-radius: 8px;
            border: 1px solid #9fb0c2;
            background: #edf4fa;
            color: var(--ink);
            font-weight: 760;
            min-height: 42px;
            box-shadow: none;
        }
        div.stButton > button:hover {
            border-color: var(--teal);
            color: var(--teal);
            background: #e2f3f1;
        }
        [data-testid="stSidebar"] div.stButton > button {
            background: rgba(255,255,255,.10) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(203,213,225,.30) !important;
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(15,118,110,.30) !important;
            border-color: rgba(103,232,249,.55) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid rgba(36,67,95,.28);
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            padding: 12px 18px;
            background: transparent;
            font-weight: 760;
        }
        .stTabs [aria-selected="true"] {
            color: var(--teal) !important;
            border-bottom: 3px solid var(--teal);
        }
        .stChatMessage {
            border-radius: 10px;
            border: 1px solid var(--line);
            background: var(--panel);
        }
        .stFileUploader section {
            border: 1px dashed #94a3b8 !important;
            background: #f1f6fb !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def header(rag_loaded: bool, hf_configured: bool) -> None:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="eyebrow">Clinical AI Demonstration</div>
          <div class="title">Neonatal ECMO Risk Dashboard</div>
          <div class="subtitle">
            Upload patient time-series data, review early-warning risk summaries, inspect visual explanation cards,
            and ask guideline-grounded questions through a RAG-enabled assistant.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Deployment profile</div>', unsafe_allow_html=True)
    labels = [
        ("web", "Web demo with inference"),
        ("local", "Local or Online mode"),
        ("rag", "Local RAG corpus"),
        ("data", "Synthetic data"),
    ]
    cols = st.columns(4)
    for i, (key, label) in enumerate(labels):
        if cols[i].button(label, use_container_width=True, key=f"profile_{key}"):
            st.session_state.profile_info = key

    info_map = {
        "web": "The public Streamlit version can call a hosted Hugging Face model for online responses. If hosted inference is unavailable, the app falls back to a template summary instead of exposing technical errors.",
        "local": "For privacy-sensitive deployment, run the same app locally and use llama3.1:8b or another local model.",
        "rag": f"The evidence layer loads a local TF-IDF RAG index from local computer. Current status: {'loaded' if rag_loaded else 'not found'}.",
        "data": "The web demo is intended for synthetic CSV files. It is not a clinically validated monitoring device.",
    }
    if st.session_state.get("profile_info"):
        st.markdown(f'<div class="info-panel">{info_map[st.session_state.profile_info]}</div>', unsafe_allow_html=True)


def card_order(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    preferred_order = ["shap_violin", "ridgeline", "next6h_gauge", "next6h_curve"]

    def card_sort_key(card: Dict[str, Any]) -> int:
        cid = str(card.get("id", "")).strip()
        return preferred_order.index(cid) if cid in preferred_order else len(preferred_order)

    return sorted(cards, key=card_sort_key)[:4]


def card_title(card: Dict[str, Any], i: int) -> str:
    cid = str(card.get("id", "")).strip()
    if cid == "shap_violin":
        return "1. Feature drivers of current risk"
    if cid == "ridgeline":
        return "2. Multi-day vital distributions"
    if cid == "next6h_gauge":
        return "3. Six-hour risk summary"
    if cid == "next6h_curve":
        return "4. Six-hour risk trajectory"
    return f"{i + 1}. {card.get('question', 'Explanation card')}"


def render_card(card: Dict[str, Any], i: int) -> None:
    """Render one explanation card without a surrounding rounded block."""
    st.markdown(f'<div class="card-title">{card_title(card, i)}</div>', unsafe_allow_html=True)
    question = str(card.get("question", "")).strip()
    if question:
        st.markdown(f'<div class="card-subtitle">{question}</div>', unsafe_allow_html=True)

    img = str(card.get("image", "")).strip()
    if img and Path(img).exists():
        st.image(img, use_container_width=True)

    with st.expander("Clinical interpretation", expanded=(i < 2)):
        st.markdown(card.get("answer", ""))

    st.markdown('<div style="height: 1.1rem;"></div>', unsafe_allow_html=True)


def handle_sidebar_question(question: str, rag: Optional[RagStore]) -> None:
    """Generate an answer from a sidebar question and append it to chat history."""
    q = (question or "").strip()
    if not q:
        return
    st.session_state.messages.append({"role": "user", "content": q})
    hits = rag.search(q, topk=RAG_TOPK) if rag else []
    prompt = build_prompt(q, st.session_state.patient_summary, hits)
    answer = generate_answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def sidebar_controls(rag: Optional[RagStore]) -> None:
    st.sidebar.title("Neonatal ECMO Demo")
    st.sidebar.caption("Research prototype · synthetic/de-identified data only")
    st.sidebar.divider()

    st.sidebar.markdown("### 1. Patient data")
    st.sidebar.caption("Upload a CSV with AR, HR, O2_sat, NIRS, SBP, DBP, pH, Lactate, or load the bundled sample.")
    uploaded = st.sidebar.file_uploader("Upload patient CSV", type=["csv"])

    c1, c2 = st.sidebar.columns(2)
    load_sample = c1.button("Load sample", use_container_width=True)
    run_risk = c2.button("Run analysis", use_container_width=True)

    if uploaded is not None:
        try:
            st.session_state.patient_df = pd.read_csv(uploaded)
            st.sidebar.success("CSV uploaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

    if load_sample:
        try:
            st.session_state.patient_df = pd.read_csv(SAMPLE_CSV)
            st.sidebar.success("Sample loaded.")
        except Exception as e:
            st.sidebar.error(f"Sample file not available: {e}")

    if run_risk:
        if st.session_state.patient_df is None:
            st.sidebar.warning("Please upload a CSV or load the sample first.")
        else:
            df = ensure_datetime_index(st.session_state.patient_df)
            risk = proxy_risk(df)
            smooth = ewma(risk)
            alarms = alarm_indices(smooth)
            p6 = six_hour_risk_summary(smooth)
            st.session_state.risk = risk
            st.session_state.smooth = smooth
            st.session_state.alarms = alarms
            st.session_state.p6 = p6
            st.session_state.patient_summary = make_patient_summary(df, smooth, p6)
            st.sidebar.success("Analysis completed.")

    st.sidebar.divider()
    st.sidebar.markdown("### 2. Current summary")
    if hasattr(st.session_state, "p6"):
        st.sidebar.metric("6h risk summary", f"{st.session_state.p6:.1%}")
        st.sidebar.metric("Alarm count", str(len(st.session_state.alarms)))
        st.sidebar.caption(f"Current smoothed risk: {float(st.session_state.smooth[-1]):.3f}")
    else:
        st.sidebar.info("Run analysis to generate patient summary.")

    st.sidebar.divider()
    st.sidebar.markdown("### 3. System status")
    hf_token = bool(get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN"))
    st.sidebar.caption(f"RAG corpus: {'loaded' if rag else 'not found'}")
    st.sidebar.caption(f"Online LLM: {'configured' if hf_token else 'HF_TOKEN missing'}")
    st.sidebar.caption(f"Mode: {'local Ollama first + HF fallback' if USE_OLLAMA else 'Hugging Face online'}")

    st.sidebar.divider()
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Dashboard overview", use_container_width=True, key="sidebar_open_dashboard"):
        st.session_state.current_view = "dashboard"
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 4. RAG assistant")
    st.sidebar.caption("Open the evidence-grounded question-answering workspace on the right.")
    if st.sidebar.button("Open RAG assistant", use_container_width=True, key="sidebar_open_chat"):
        st.session_state.current_view = "chat"
        st.rerun()


# ====================== Main app ======================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_css()

    rag = load_rag_store(str(CORPUS_DIR))
    cards = load_cards(str(PRECOMPUTED_CARDS))
    hf_configured = bool(get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN"))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_df" not in st.session_state:
        st.session_state.patient_df = None
    if "patient_summary" not in st.session_state:
        st.session_state.patient_summary = "No patient data loaded yet."
    if "profile_info" not in st.session_state:
        st.session_state.profile_info = ""

    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"

    sidebar_controls(rag)
    header(rag_loaded=bool(rag), hf_configured=hf_configured)

    if st.session_state.current_view == "chat":
        st.markdown('<div class="section-kicker">Evidence-grounded question answering</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="primary-note">Ask about the uploaded patient data or ECMO guideline evidence.</div>',
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Example: What are the main risk drivers and what should the team monitor next?")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Generating RAG-based answer…"):
                    hits = rag.search(question, topk=RAG_TOPK) if rag else []
                    prompt = build_prompt(question, st.session_state.patient_summary, hits)
                    answer = generate_answer(prompt)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        st.markdown('<div class="section-kicker">Quick prompts</div>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        examples = [
            "Summarize the current risk status in plain language.",
            "Which signals are most concerning right now?",
            "What should the bedside team monitor over the next hour?",
        ]
        for i, q in enumerate(examples):
            if ex_cols[i].button(q, use_container_width=True, key=f"chat_quick_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                hits = rag.search(q, topk=RAG_TOPK) if rag else []
                prompt = build_prompt(q, st.session_state.patient_summary, hits)
                answer = generate_answer(prompt)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

        if st.button("Clear chat", key="main_clear_chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.markdown('<div class="section-kicker">Risk review</div>', unsafe_allow_html=True)
        if st.session_state.patient_df is not None and hasattr(st.session_state, "smooth"):
            cols = st.columns([0.27, 0.27, 0.46])
            cols[0].metric("Six-hour risk summary", f"{st.session_state.p6:.1%}")
            cols[1].metric("Alarm count", str(len(st.session_state.alarms)))
            cols[2].markdown(
                '<div class="primary-note">The public demo uses a lightweight preconfigured risk summarizer. It is intended for demonstration, not clinical validation.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                risk_plot(st.session_state.patient_df, st.session_state.risk, st.session_state.smooth, st.session_state.alarms),
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="primary-note">Load the synthetic sample or upload a patient CSV from the sidebar, then run analysis to generate the risk trajectory.</div>',
                unsafe_allow_html=True,
            )

        if cards:
            st.markdown('<div class="section-kicker">Visual explanation cards</div>', unsafe_allow_html=True)
            cards_sorted = card_order(cards)
            for row_start in range(0, len(cards_sorted), 2):
                row_cols = st.columns(2, gap="large")
                for j in range(2):
                    idx = row_start + j
                    if idx >= len(cards_sorted):
                        break
                    with row_cols[j]:
                        render_card(cards_sorted[idx], idx)


if __name__ == "__main__":
    main()
