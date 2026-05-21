# Streamlit chat UI + RAG + Precomputed Patient Summary Cards
#
# Run:
#   streamlit run app_chat.py

from __future__ import annotations

import json
import os
import pickle
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
# These are intentionally not exposed in the UI.
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
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
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
    """Lightweight fallback risk score for web demo uploads."""
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
        f"- Time window summarized: last ~30 minutes",
        f"- Current risk index: {float(smooth[-1]):.3f}" if smooth is not None and len(smooth) else "- Current risk index: unavailable",
        f"- Six-hour risk summary: {p6:.1%}" if p6 is not None else "- Six-hour risk summary: unavailable",
        f"- Active flags: {', '.join(flags) if flags else 'none detected by demo thresholds'}",
        f"- Vitals/labs statistics: {json.dumps(stats, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


# ====================== LLM generation ======================
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


def _extract_section(text: str, start_label: str, end_labels: List[str]) -> str:
    """Small helper for the offline fallback answer."""
    if start_label not in text:
        return ""
    part = text.split(start_label, 1)[1]
    for lab in end_labels:
        if lab in part:
            part = part.split(lab, 1)[0]
    return part.strip()


def offline_summary_answer(prompt: str) -> str:
    """User-facing fallback when hosted/local LLM is unavailable.

    This intentionally hides low-level Hugging Face/provider errors from demo users
    and still gives a useful, cautious summary using the prompt content that has
    already been assembled from patient data and retrieved RAG snippets.
    """
    user_q = _extract_section(
        prompt,
        "User question:",
        ["Patient summary:", "Retrieved ECMO evidence snippets:", "Please answer"],
    )
    patient = _extract_section(
        prompt,
        "Patient summary:",
        ["Retrieved ECMO evidence snippets:", "Please answer"],
    )
    evidence = _extract_section(
        prompt,
        "Retrieved ECMO evidence snippets:",
        ["Please answer"],
    )

    patient_short = patient if patient else "No patient summary is currently available. Please upload or load a CSV and run analysis first."
    evidence_short = evidence if evidence and "No RAG evidence retrieved" not in evidence else "No guideline snippets were retrieved for this query."
    if len(patient_short) > 1200:
        patient_short = patient_short[:1200].rstrip() + "..."
    if len(evidence_short) > 900:
        evidence_short = evidence_short[:900].rstrip() + "..."

    return (
        "LLM offline. Based on the available patient summary and local evidence, here is a template-based summary.\n\n"
        f"**Question:** {user_q or 'General ECMO risk question'}\n\n"
        "**Patient summary**\n"
        f"{patient_short}\n\n"
        "**Evidence context**\n"
        f"{evidence_short}\n\n"
        "**Suggested monitoring focus**\n"
        "- Review oxygenation signals such as O2 saturation, PaO2, and NIRS for sustained decline rather than isolated noise.\n"
        "- Check hemodynamic stability, especially systolic/diastolic blood pressure and heart-rate trends.\n"
        "- Pay attention to metabolic stress markers such as lactate, pH, and base excess when available.\n"
        "- If the risk trajectory is rising or multiple red flags appear together, increase bedside review frequency and prepare escalation discussion.\n"
        "- This is a demo-oriented summary and should not be used as validated clinical advice."
    )


def call_huggingface(prompt: str, timeout: int = 120) -> str:
    token = get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN")
    model = get_secret("HF_MODEL", DEFAULT_HF_MODEL)
    provider = get_secret("HF_PROVIDER", DEFAULT_HF_PROVIDER)

    # Do not expose configuration/provider errors to end users in the public demo.
    if not token:
        return offline_summary_answer(prompt)

    try:
        from huggingface_hub import InferenceClient
    except Exception:
        return offline_summary_answer(prompt)

    try:
        # Prefer OpenAI-compatible chat-completion style. This avoids calling
        # text_generation on conversational-only hosted models.
        if provider and provider.lower() != "auto":
            client = InferenceClient(provider=provider, api_key=token, timeout=timeout)
        else:
            client = InferenceClient(api_key=token, timeout=timeout)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful clinical decision-support demo assistant. "
                    "Use the patient summary and retrieved ECMO evidence. "
                    "Do not diagnose. Give concise monitoring-oriented suggestions."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Newer huggingface_hub API.
        try:
            out = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=550,
                temperature=0.25,
            )
            content = out.choices[0].message.content
            if content:
                return str(content).strip()
        except Exception:
            pass

        # Older huggingface_hub API.
        try:
            old_client = InferenceClient(model=model, token=token, timeout=timeout)
            out_old = old_client.chat_completion(
                messages=messages,
                max_tokens=550,
                temperature=0.25,
                top_p=0.9,
            )
            content_old = out_old.choices[0].message.content
            if content_old:
                return str(content_old).strip()
        except Exception:
            pass

    except Exception:
        pass

    return offline_summary_answer(prompt)


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
    fig.add_trace(go.Scatter(x=x, y=risk, mode="lines", name="Raw risk", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", name="Smoothed risk", line=dict(width=3)))
    if alarms:
        ai = [i for i in alarms if i < len(smooth)]
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in ai],
                y=[smooth[i] for i in ai],
                mode="markers",
                name="Alarms",
                marker=dict(size=10, symbol="diamond"),
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=12, r=12, t=36, b=12),
        title="Recent Early-Warning Risk Trajectory",
        yaxis=dict(range=[0, max(0.2, min(1.0, float(np.nanmax(smooth)) * 1.35 if len(smooth) else 0.2))]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important; height: 0px !important;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1240px;}
        .hero {
            border: 1px solid rgba(148,163,184,.28);
            border-radius: 28px;
            padding: 26px 28px;
            background:
              radial-gradient(circle at 8% 8%, rgba(14,165,233,.18), transparent 28%),
              radial-gradient(circle at 90% 10%, rgba(99,102,241,.16), transparent 30%),
              linear-gradient(135deg, rgba(255,255,255,.92), rgba(248,250,252,.84));
            box-shadow: 0 18px 50px rgba(15,23,42,.08);
            margin-bottom: 18px;
        }
        .hero-title {font-size: 2.25rem; font-weight: 900; letter-spacing: -0.03em; margin: 0;}
        .hero-sub {font-size: 1.02rem; line-height: 1.55; color: rgba(15,23,42,.68); margin-top: 8px; max-width: 900px;}
        .badge-row {display:flex; gap:8px; flex-wrap:wrap; margin-top: 16px;}
        .badge {
            border: 1px solid rgba(148,163,184,.35);
            border-radius: 999px; padding: 7px 11px;
            background: rgba(255,255,255,.72); font-weight: 750; font-size: .88rem;
            color: rgba(15,23,42,.78);
        }
        .soft-card {
            border: 1px solid rgba(148,163,184,.30);
            border-radius: 22px;
            padding: 18px;
            background: rgba(255,255,255,.82);
            box-shadow: 0 10px 30px rgba(15,23,42,.06);
        }
        .section-label {font-weight: 850; font-size: 1.15rem; margin: 0 0 8px 0;}
        .muted {color: rgba(15,23,42,.60); font-size: .92rem; line-height: 1.45;}
        div.stButton > button {
            border-radius: 999px; border: 1px solid rgba(14,165,233,.35);
            background: linear-gradient(135deg, rgba(14,165,233,.12), rgba(99,102,241,.10));
            font-weight: 800; min-height: 42px;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px; padding: 8px 16px; background: rgba(248,250,252,.8);
        }
        .stChatMessage {border-radius: 18px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">🫀 Neonatal ECMO Risk Dashboard</div>
          <div class="hero-sub">
            Upload patient time-series data, review early-warning risk trends, and ask guideline-grounded questions through an online Hugging Face LLM or optional local Ollama deployment.
          </div>
          <div class="badge-row">
            <span class="badge">📈 Early-warning trajectory</span>
            <span class="badge">🔎 Local RAG corpus</span>
            <span class="badge">💬 Online LLM answers</span>
            <span class="badge">🔒 Local deployment ready</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====================== Main app ======================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    hero()

    rag = load_rag_store(str(CORPUS_DIR))
    cards = load_cards(str(PRECOMPUTED_CARDS))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_df" not in st.session_state:
        st.session_state.patient_df = None
    if "patient_summary" not in st.session_state:
        st.session_state.patient_summary = "No patient data loaded yet."

    left, right = st.columns([0.36, 0.64], gap="large")

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">1) Patient Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Upload a CSV with columns such as AR, HR, O2_sat, NIRS, SBP, DBP, pH, Lactate. Or load the included sample.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload patient CSV", type=["csv"], label_visibility="collapsed")
        c1, c2 = st.columns(2)
        load_sample = c1.button("Load sample", use_container_width=True)
        run_risk = c2.button("Run analysis", use_container_width=True)

        if uploaded is not None:
            try:
                st.session_state.patient_df = pd.read_csv(uploaded)
                st.success("CSV uploaded.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        if load_sample:
            try:
                st.session_state.patient_df = pd.read_csv(SAMPLE_CSV)
                st.success("Sample loaded.")
            except Exception as e:
                st.error(f"Sample file not available: {e}")

        if run_risk:
            if st.session_state.patient_df is None:
                st.warning("Please upload a CSV or load the sample first.")
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
                st.success("Analysis completed.")

        st.markdown("---")
        st.markdown("**System status**")
        hf_token = bool(get_secret("HF_TOKEN") or get_secret("HUGGINGFACEHUB_API_TOKEN"))
        st.caption(f"RAG corpus: {'loaded' if rag else 'not found'}")
        st.caption(f"Online LLM: {'configured' if hf_token else 'HF_TOKEN missing'}")
        st.caption(f"Mode: {'local Ollama first + HF fallback' if USE_OLLAMA else 'Hugging Face online'}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">2) Current Summary</div>', unsafe_allow_html=True)
        if hasattr(st.session_state, "p6"):
            m1, m2 = st.columns(2)
            m1.metric("6h Risk Summary", f"{st.session_state.p6:.1%}")
            m2.metric("Alarms", str(len(st.session_state.alarms)))
            st.caption(f"Current smoothed risk: {float(st.session_state.smooth[-1]):.3f}")
        else:
            st.info("Run analysis to generate patient risk summary.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        tab_dash, tab_chat = st.tabs(["📊 Dashboard", "💬 RAG Chat"])

        with tab_dash:
            if st.session_state.patient_df is not None and hasattr(st.session_state, "smooth"):
                st.plotly_chart(
                    risk_plot(st.session_state.patient_df, st.session_state.risk, st.session_state.smooth, st.session_state.alarms),
                    use_container_width=True,
                )
            else:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.markdown("### Welcome")
                st.write("Upload a patient CSV or load the sample, then click **Run analysis** to generate the risk trajectory.")
                st.markdown("</div>", unsafe_allow_html=True)

            if cards:
                st.markdown("### Visual Explanation Cards")
                grid = st.columns(2, gap="large")
                for i, card in enumerate(cards[:4]):
                    with grid[i % 2]:
                        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                        st.markdown(f"**{card.get('question', 'Explanation card')}**")
                        img = str(card.get("image", "")).strip()
                        if img and Path(img).exists():
                            st.image(img, use_container_width=True)
                        with st.expander("View explanation", expanded=(i < 2)):
                            st.markdown(card.get("answer", ""))
                        st.markdown("</div>", unsafe_allow_html=True)

        with tab_chat:
            st.markdown("Ask questions about the uploaded patient data or ECMO guideline evidence.")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            question = st.chat_input("Example: What are the main risk drivers and what should the team monitor next?")
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Calling online LLM…"):
                        hits = rag.search(question, topk=RAG_TOPK) if rag else []
                        prompt = build_prompt(question, st.session_state.patient_summary, hits)
                        answer = generate_answer(prompt)
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            ex_cols = st.columns(3)
            examples = [
                "Summarize the current risk status in plain language.",
                "Which signals are most concerning right now?",
                "What should the bedside team monitor over the next hour?",
            ]
            for i, q in enumerate(examples):
                if ex_cols[i].button(q, use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    hits = rag.search(q, topk=RAG_TOPK) if rag else []
                    prompt = build_prompt(q, st.session_state.patient_summary, hits)
                    answer = generate_answer(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

            if st.button("Clear chat"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()


