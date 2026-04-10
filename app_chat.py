# Streamlit chat UI + RAG + Precomputed Patient Summary Cards
#
# Run:
#   streamlit run app_chat.py

from __future__ import annotations
import os, json, requests, pickle
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ====================== small helpers ======================

def _np(x):
    return np.asarray(x, dtype=float)

def _gauss_kernel(k: int) -> np.ndarray:
    k = max(3, int(k) | 1)  # odd
    x = np.linspace(-2.5, 2.5, k)
    g = np.exp(-0.5 * x * x)
    g /= g.sum()
    return g

def _smooth_series(y, k: int):
    y = _np(y).ravel()
    if y.size < 3:
        return y
    g = _gauss_kernel(k)
    pad = k // 2
    ypad = np.r_[y[pad:0:-1], y, y[-2:-pad-2:-1]]
    z = np.convolve(ypad, g, mode="valid")
    return z

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "AR" in df.columns:
        d = df.copy()
        d["AR"] = pd.to_datetime(d["AR"])
        d = d.set_index("AR").sort_index()
        return d
    return df.copy()

def _nan_to(x, val=0.0):
    x = np.asarray(x, float)
    x[~np.isfinite(x)] = val
    return x


# ====================== RAG ======================

_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


class RagStore:
    def __init__(self, store_dir: Path):
        self.dir = store_dir
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X: Optional[sp.csr_matrix] = None
        self.E: Optional[np.ndarray] = None
        self._dense_model: Optional["SentenceTransformer"] = None

    def load(self, use_dense: bool = True, dense_model: str = "BAAI/bge-small-en-v1.5"):
        texts_fp = self.dir / "texts.jsonl"
        metas_fp = self.dir / "metas.jsonl"
        vect_fp = self.dir / "vectorizer.pkl"
        xtfidf_fp = self.dir / "X_tfidf.npz"
        if not texts_fp.exists() or not metas_fp.exists() or not vect_fp.exists() or not xtfidf_fp.exists():
            raise FileNotFoundError("Missing corpus files")

        with texts_fp.open("r", encoding="utf-8") as f:
            self.texts = []
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj["text"] if isinstance(obj, dict) and "text" in obj else obj)

        with metas_fp.open("r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f]
        self.metas = [r["meta"] if isinstance(r, dict) and "meta" in r else r for r in raw]

        with vect_fp.open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.X = sp.load_npz(xtfidf_fp).tocsr()

        e_path = self.dir / "E_dense.npy"
        if use_dense and _HAS_ST and e_path.exists():
            self.E = np.load(e_path)
            try:
                self._dense_model = SentenceTransformer(dense_model)
            except Exception:
                self._dense_model = None
        else:
            self.E = None
            self._dense_model = None
        return self

    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo + 1e-9)

    def search(self, query: str, topk: int = 8, alpha_lex: float = 0.7, beta_dense: float = 0.3) -> List[Dict[str, Any]]:
        assert self.vectorizer is not None and self.X is not None
        qv = self.vectorizer.transform([query])
        s_lex = (self.X @ qv.T).toarray().ravel()
        s_lex = self._minmax01(s_lex)

        if self._dense_model is not None and self.E is not None:
            qvec = self._dense_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
            s_dense = (self.E @ qvec) / (np.linalg.norm(qvec) + 1e-9)
            s_dense = self._minmax01(s_dense)
        else:
            s_dense = np.zeros_like(s_lex)

        fused = alpha_lex * s_lex + beta_dense * s_dense
        order = np.argsort(-fused)[:topk]
        rows = []
        for i, idx in enumerate(order, start=1):
            rows.append({
                "rank": i,
                "score": float(fused[idx]),
                "text": self.texts[idx],
                "meta": self.metas[idx],
            })
        return rows


# ====================== LLM via Ollama ======================
def ollama_generate(model: str, prompt: str, base_url: str, timeout: int = 120) -> Optional[str]:
    url = f"{base_url.rstrip('/')}/api/generate"
    try:
        resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass

    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        if hf_token:
            from huggingface_hub import InferenceClient
            client = InferenceClient(hf_model, token=hf_token)
            out = client.text_generation(
                prompt,
                max_new_tokens=256,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.05,
            )
            return (out or "").strip()
    except Exception:
        pass

    try:
        return "LLM offline. Based on patient summary and internal evidence, here are key points:\n- Ensure adequate oxygenation and blood pressure.\n- Consider checking recent lab trends.\n- If risk trajectory is rising, shorten monitoring interval and prepare escalation."
    except Exception:
        return None


# ====================== Risk, Alarms, Forecast core ======================
def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def proxy_risk_v2(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    x_terms = []

    def ffill(s, lim=240):
        return pd.to_numeric(s, errors="coerce").ffill(limit=lim)

    if "O2_sat" in df.columns:
        o2 = ffill(df["O2_sat"])
        x_terms.append(0.45 * _sig((72 - o2) / 4.0))
    if "SBP" in df.columns:
        sbp = ffill(df["SBP"])
        x_terms.append(0.35 * _sig((58 - sbp) / 4.0))
    if "DBP" in df.columns:
        dbp = ffill(df["DBP"])
        x_terms.append(0.20 * _sig((32 - dbp) / 3.0))
    if "HR" in df.columns:
        hr = ffill(df["HR"])
        x_terms.append(0.15 * np.maximum(_sig((hr - 190) / 6.0), _sig((60 - hr) / 6.0)))

    if not x_terms:
        return np.zeros(n, dtype=float)

    X = np.vstack(x_terms)
    risk = 1.0 - np.prod(1.0 - X, axis=0)
    return np.clip(risk, 0.0, 1.0)

def ewma_alarm(risk: np.ndarray, step_sec=15, alpha=0.35,
               threshold=0.75, hold_min=1.0, refractory_min=10.0):
    s = 0.0
    above = 0
    hold_steps = max(1, int(hold_min * 60 / step_sec))
    ref_steps = max(1, int(refractory_min * 60 / step_sec))
    alarms: List[int] = []
    smoothed = np.zeros_like(risk, dtype=float)
    t = 0
    while t < len(risk):
        s = alpha * risk[t] + (1 - alpha) * s
        smoothed[t] = s
        if s >= threshold:
            above += 1
            if above >= hold_steps:
                alarms.append(t)
                t += ref_steps
                above = 0
                continue
        else:
            above = 0
        t += 1
    return {"alarms": alarms, "smoothed": smoothed}

def predict_p6h_prob(smooth_like, step_sec: int = 15,
                     calib_path: str = "out/calib_6h_lr.json") -> float:
    s = _np(smooth_like).ravel()
    if s.size < 8:
        return 0.02
    r_now = float(np.clip(s[-1], 0.0, 1.0))
    w1h = max(2, int(round(3600 / step_sec)))
    seg = s[-w1h:] if s.size >= w1h else s
    r_mean = float(np.clip(np.nanmean(seg), 0.0, 1.0))
    r_max = float(np.clip(np.nanmax(seg), 0.0, 1.0))

    k = max(2, int(round(1800 / step_sec)))
    x = np.arange(min(k, s.size), dtype=float)
    x -= x.mean()
    sl30 = float(((x * s[-len(x):]).sum()) / (x * x).sum()) if s.size >= 2 else 0.0

    w = np.array([2.2, 1.4, 1.0, 0.8], dtype=float)
    b = -2.0
    try:
        if os.path.exists(calib_path):
            blob = json.load(open(calib_path, "r", encoding="utf-8"))
            if "w" in blob and "b" in blob:
                w = _np(blob["w"])
                b = float(blob["b"])
            elif "coef" in blob:
                c = blob["coef"]
                w = _np([
                    c.get("r_now", 2.2),
                    c.get("r_mean", 1.4),
                    c.get("r_max", 1.0),
                    c.get("slope30_pos", 0.8),
                ])
                b = float(blob.get("bias", -2.0))
    except Exception:
        pass

    z = float(b + np.dot(w, np.array([r_now, r_mean, r_max, max(0.0, sl30)], dtype=float)))
    return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.001, 0.99))


# ====================== Precomputed Cards ======================

@st.cache_data(show_spinner=False)
def load_precomputed_cards(path: str = "precomputed_cards.json") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for i, c in enumerate(data):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id", f"card_{i}")).strip()
            q = str(c.get("question", "")).strip()
            a = str(c.get("answer", "")).strip()
            img = str(c.get("image", "")).strip()
            if not q or not a:
                continue
            cleaned.append(
                {
                    "id": cid,
                    "question": q,
                    "answer": a,
                    "image": img,
                }
            )
        return cleaned
    except Exception as e:
        st.warning(f"Failed to load precomputed_cards.json: {e}")
        return []

# ====================== Prompt builder ======================

SYSTEM_MSG = (
    "You are a careful clinical assistant. Use the single-ventricle infants patient summary first, "
    "then internal evidence to answer the user succinctly. Provide actionable guidance. "
    "If evidence is mixed or uncertain, say so clearly."
)

def build_prompt_no_citations(user_question: str,
                              patient_summary: str,
                              snippets: List[Dict[str, Any]]) -> str:
    ctx_text = "\n\n".join(s["text"] for s in snippets)
    return (
        f"{SYSTEM_MSG}\n\n"
        f"Patient summary:\n{patient_summary}\n\n"
        f"User question:\n{user_question}\n\n"
        f"Internal evidence (use for reasoning; do NOT cite):\n{ctx_text}\n\n"
        f"Answer:"
    )


# ====================== UI ======================

st.set_page_config(
    page_title="ECMO RAG Chat + Early Warning",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ====================== HIDE STREAMLIT TOOLBAR / ADVANCED ======================
st.markdown(
    """
    <style>
      /* Hide Streamlit toolbar (the 'Advanced' menu / gear) */
      [data-testid="stToolbar"] {visibility: hidden !important; height: 0px !important;}
      /* Hide hamburger menu in top-right (optional) */
      #MainMenu {visibility: hidden;}
      /* Hide "Made with Streamlit" footer (optional) */
      footer {visibility: hidden;}
      /* Reduce extra top padding after hiding toolbar */
      .block-container { padding-top: 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Enlarge font in chat messages ----
st.markdown(
    """
    <style>
    /* Make user+assistant chat text larger */
    .stChatMessage .stMarkdown p {
        font-size: 1.6rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== HERO / BRANDING ======================

st.markdown(
    """
    <style>
      .hero {
        border: 1px solid rgba(148,163,184,.35);
        border-radius: 22px;
        padding: 18px 20px;
        background: linear-gradient(135deg, rgba(14,165,233,.12), rgba(99,102,241,.10), rgba(16,185,129,.10));
        box-shadow: 0 8px 24px rgba(2,6,23,.05);
        margin-bottom: 14px;
      }
      .hero-top{
        display:flex; align-items:center; justify-content:space-between; gap:12px;
      }
      .hero-title{
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
      }
      .hero-sub{
        margin: 6px 0 0 0;
        color: rgba(15,23,42,.75);
        font-size: 1.02rem;
        line-height: 1.35;
      }
      .pill{
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.45);
        background: rgba(255,255,255,.55);
        font-weight: 700;
        font-size: .95rem;
        white-space: nowrap;
      }
      .quickgrid{
        display:grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }
      .quickcard{
        border: 1px solid rgba(148,163,184,.35);
        border-radius: 16px;
        padding: 12px 12px;
        background: rgba(255,255,255,.58);
      }
      .quickcard .k{
        font-weight: 800;
        margin-bottom: 4px;
      }
      .quickcard .v{
        color: rgba(15,23,42,.72);
        font-size: .95rem;
        line-height: 1.3;
      }
      @media (max-width: 900px){
        .quickgrid{ grid-template-columns: 1fr; }
      }
    </style>

    <div class="hero">
      <div class="hero-top">
        <div>
          <div class="hero-title">🫀 Neonatal ECMO Risk Dashboard</div>
          <div class="hero-sub">
            Combine <b>patient time-series</b> + <b>early-warning risk</b> + <b>RAG guideline evidence</b> to support bedside decisions (single-ventricle focused).
          </div>
        </div>
        <div class="pill">🧠 RAG + 📈 Early Warning + 💬 Chat</div>
      </div>

      <div class="quickgrid">
        <div class="quickcard">
          <div class="k">1) Upload / Load</div>
          <div class="v">Upload infant vitals/labs CSV (or load sample) to generate risk & alarms.</div>
        </div>
        <div class="quickcard">
          <div class="k">2) Review Cards</div>
          <div class="v">See next-6h probability, trajectory, explanations, and distributions at a glance.</div>
        </div>
        <div class="quickcard">
          <div class="k">3) Ask the Chat</div>
          <div class="v">Ask for concise guidance grounded in patient summary + internal evidence snippets.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ====================== CUSTOM TOP BAR (fills left blank) ======================
st.markdown(
    """
    <style>
      .topbar{
        display:flex; align-items:center; justify-content:space-between;
        gap:12px; margin: 0.2rem 0 0.8rem 0;
      }
      .brand{
        display:flex; align-items:center; gap:12px;
      }
      .brand h1{
        font-size: 2.05rem; font-weight: 850; margin:0; line-height:1.05;
      }
      .brand .sub{
        margin-top: 0.25rem;
        color: rgba(15,23,42,.65);
        font-size: 0.95rem;
      }
      .pillrow{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
      .pill{
        border: 1px solid rgba(148,163,184,.40);
        background: rgba(255,255,255,.65);
        padding: 7px 10px; border-radius: 999px;
        font-weight: 700; font-size: 0.88rem;
        display:inline-flex; align-items:center; gap:8px;
      }
      .logoBox{
        width: 44px; height: 44px; border-radius: 14px;
        background: linear-gradient(135deg, rgba(14,165,233,.20), rgba(99,102,241,.18), rgba(16,185,129,.18));
        border: 1px solid rgba(148,163,184,.35);
        display:flex; align-items:center; justify-content:center;
        box-shadow: 0 8px 18px rgba(2,6,23,.06);
        flex: 0 0 auto;
      }
      @media(max-width: 900px){
        .topbar{ flex-direction: column; align-items:flex-start; }
        .pillrow{ justify-content:flex-start; }
      }
    </style>

    <div class="topbar">
      <div class="brand">
        <div class="logoBox">🫀</div>
        <div>
          <h1>Neonatal ECMO: RAG Chat + Early Warning</h1>
          <div class="sub">Single-ventricle focused • Patient risk trajectory • Guideline-grounded suggestions</div>
        </div>
      </div>
      <div class="pillrow">
        <div class="pill">🧠 RAG Evidence</div>
        <div class="pill">📈 Early Warning</div>
        <div class="pill">💬 Clinician Chat</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Title + Advanced
c_gear, c_title = st.columns([0.10, 0.90])

corpus_dir = str(Path("./store_txt_rag").resolve())
use_dense = True
dense_model = "BAAI/bge-small-en-v1.5"
default_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_base_url = default_base
ollama_model = "llama3.1:8b"
topk = 8
alpha = 0.50
thr = 0.50
hold = 1.0
refr = 10.0

with c_gear:
    st.markdown(
        """
        <style>
        .equal-btn .stButton>button{
            height:44px;
            padding:0 18px;
            border-radius:999px;
            border:1px solid #d1d5db;
            background:linear-gradient(180deg,#fff,#f3f4f6);
            color:#111827;
            font-weight:600
        }
        .equal-btn .stButton>button:hover{
            border-color:#9ca3af;
            background:linear-gradient(180deg,#f9fafb,#e5e7eb)
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.popover("⚙️ Advanced", use_container_width=True):
        st.subheader("Advanced")
        corpus_dir = st.text_input("Corpus directory", value=corpus_dir)
        use_dense = st.checkbox("Use dense embeddings (if available)", value=use_dense)
        dense_model = st.text_input("Dense model", value=dense_model)

        st.markdown("---")
        st.caption("Generation")
        ollama_base_url = st.text_input("Ollama base URL", value=ollama_base_url)
        ollama_model = st.text_input("Ollama model", value=ollama_model)
        topk = st.slider("Internal top-k chunks", 3, 12, topk)

        st.markdown("---")
        st.caption("Alarm parameters")
        alpha = st.slider("EWMA α", 0.05, 0.95, alpha, 0.05)
        thr = st.slider("Threshold", 0.10, 0.95, thr, 0.01)
        hold = st.slider("Hold time (min)", 0.5, 5.0, hold, 0.5)
        refr = st.slider("Refractory (min)", 1.0, 30.0, refr, 1.0)

# Load RAG once per settings
reload_needed = (
    "RAG" not in st.session_state
    or st.session_state.get("rag_path") != corpus_dir
    or st.session_state.get("rag_dense") != (use_dense and _HAS_ST)
)

if reload_needed:
    try:
        st.session_state["RAG"] = RagStore(Path(corpus_dir)).load(
            use_dense=use_dense and _HAS_ST,
            dense_model=dense_model,
        )
        st.session_state["rag_path"] = corpus_dir
        st.session_state["rag_dense"] = (use_dense and _HAS_ST)
        st.success("Corpus loaded.")
    except Exception as e:
        st.error(f"Failed to load corpus: {e}")

rag: Optional[RagStore] = st.session_state.get("RAG")

# ====================== Patient Data ======================

st.header("Patient Data")
c_up, c_load, c_run = st.columns([0.65, 0.18, 0.17])

with c_up:
    up = st.file_uploader(
        "Upload CSV with vitals/labs (time col 'AR' optional)",
        type=["csv"],
        label_visibility="collapsed",
    )

with c_load:
    st.markdown('<div class="equal-btn">', unsafe_allow_html=True)
    load_clicked = st.button("Load sample", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c_run:
    st.markdown('<div class="equal-btn">', unsafe_allow_html=True)
    run_clicked = st.button("Run risk + alarm", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if load_clicked:
    try:
        st.session_state["patient_df"] = pd.read_csv("sim_ecmo_timeseries.csv")
        st.success("Sample loaded.")
    except Exception as e:
        st.error(f"Failed to read sample: {e}")

if up is not None:
    try:
        st.session_state["patient_df"] = pd.read_csv(up)
        st.success("CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if run_clicked and "patient_df" in st.session_state:
    try:
        df_i = _ensure_dt_index(st.session_state["patient_df"])
        r = proxy_risk_v2(df_i)
        res = ewma_alarm(
            r,
            step_sec=15,
            alpha=alpha,
            threshold=thr,
            hold_min=hold,
            refractory_min=refr,
        )
        st.session_state["risk"] = _np(r).ravel()
        st.session_state["smooth"] = _np(res["smoothed"]).ravel()
        st.session_state["alarms"] = list(res["alarms"])

        p6 = predict_p6h_prob(
            st.session_state["smooth"],
            step_sec=15,
            calib_path="out/calib_6h_lr.json",
        )
        st.session_state["p6h"] = p6

        tail = df_i.tail(120)
        flags = []
        last = tail.iloc[-1]
        if last.get("O2_sat", 100) < 70:
            flags.append("Low SpO2")
        if last.get("SBP", 200) < 55:
            flags.append("Low SBP")
        if last.get("DBP", 200) < 30:
            flags.append("Low DBP")
        if last.get("HR", 0) < 60:
            flags.append("Bradycardia")
        if last.get("HR", 0) > 190:
            flags.append("Tachycardia")

        base_cols = ["HR", "RR", "O2_sat", "NIRS", "SBP", "DBP"]
        cols = [c for c in base_cols if c in tail.columns]
        stats = {
            c: {
                "mean": round(pd.to_numeric(tail[c], errors="coerce").mean(), 2),
                "min": round(pd.to_numeric(tail[c], errors="coerce").min(), 2),
                "max": round(pd.to_numeric(tail[c], errors="coerce").max(), 2),
            }
            for c in cols
        }
        r_now = round(float(st.session_state["smooth"][-1]), 3)

        st.session_state["patient_summary"] = "\n".join(
            [
                "Recent patient summary (~30 min):",
                f"Flags: {', '.join(flags) if flags else 'none'}",
                f"6h event probability: {p6:.1%}",
                f"Current risk (smoothed): {r_now}",
                f"Stats (mean/min/max): {stats}",
            ]
        )

        st.success("Risk + alarm computed.")
    except Exception as e:
        st.error(f"Alarm/forecast failed: {e}")

# ====================== Chat state ======================

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "patient_summary" not in st.session_state:
    st.session_state["patient_summary"] = "(no patient uploaded)"

tab_cards, tab_chat = st.tabs(["📊 Neonatal ECMO Dashboard", "💬 Ask ECMO (RAG Chat)"])


# ========== Tab 1: Precomputed Summary Cards ==========

with tab_cards:
    st.markdown("### 📊 Patient Visual Summary Cards")
    st.caption(
        "Curated offline cards built from model outputs. "
        "Review them quickly like a clinical dashboard."
    )

    cards = load_precomputed_cards()

    if not cards:
        st.info(
            "No `precomputed_cards.json` detected. "
            "Please generate precomputed cards and figures in the project folder."
        )
    else:
        preferred_order = [
            "next6h_gauge",
            "next6h_curve",
            "shap_violin",
            "ridgeline",
        ]

        def sort_key(c: Dict[str, Any]) -> int:
            cid = c.get("id", "")
            return preferred_order.index(cid) if cid in preferred_order else len(preferred_order)

        cards_sorted = sorted(cards, key=sort_key)

        # --- nice card styling ---
        st.markdown(
            """
            <style>
              .dashcard{
                border: 1px solid rgba(148,163,184,.35);
                border-radius: 18px;
                padding: 14px 14px;
                background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(248,250,252,.85));
                box-shadow: 0 8px 24px rgba(2,6,23,.05);
                margin-bottom: 14px;
              }
              .dashcard h4{
                margin: 0 0 6px 0;
                font-size: 1.15rem;
              }
              .dashmeta{
                color: rgba(15,23,42,.65);
                font-size: .92rem;
                margin-bottom: 10px;
              }
              .dashdivider{
                height:1px; background: rgba(148,163,184,.25); margin: 10px 0 10px 0;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Optional: small overview metrics row if already computed
        c1, c2, c3, c4 = st.columns(4)
        p6 = st.session_state.get("p6h", None)
        alarms = st.session_state.get("alarms", [])
        smooth = st.session_state.get("smooth", None)
        if p6 is not None:
            c1.metric("Next 6h Probability", f"{p6*100:.1f}%")
        else:
            c1.metric("Next 6h Probability", "—")
        c2.metric("Alarms (count)", f"{len(alarms)}")
        if smooth is not None and len(smooth) > 0:
            c3.metric("Current Smoothed Risk", f"{float(smooth[-1]):.3f}")
        else:
            c3.metric("Current Smoothed Risk", "—")
        c4.metric("RAG Corpus", "Loaded" if st.session_state.get("RAG") else "—")

        st.markdown("")

        # --- grid layout: 2 columns ---
        cols = st.columns(2, gap="large")

        def pretty_title(cid: str) -> str:
            if cid == "next6h_gauge":
                return "1️⃣ Next 6h event probability"
            if cid == "next6h_curve":
                return "2️⃣ Next 6h danger trajectory"
            if cid == "shap_violin":
                return "3️⃣ Why this risk? (feature drivers)"
            if cid == "ridgeline":
                return "4️⃣ Multi-day vital distributions"
            return cid or "Card"

        for i, card in enumerate(cards_sorted):
            col = cols[i % 2]
            with col:
                cid = str(card.get("id", "")).strip()
                title = pretty_title(cid)

                st.markdown(f'<div class="dashcard">', unsafe_allow_html=True)
                st.markdown(f"#### {title}")
                st.markdown(
                    f'<div class="dashmeta">Neonatal ECMO • Visual Summary • Offline curated</div>',
                    unsafe_allow_html=True,
                )

                # Use expander to keep page readable but still "same page"
                with st.expander("📝 Question & Explanation", expanded=True if i < 2 else False):
                    st.markdown("**Question**")
                    st.markdown(card.get("question", ""))
                    st.markdown('<div class="dashdivider"></div>', unsafe_allow_html=True)
                    st.markdown("**Explanation**")
                    st.markdown(card.get("answer", ""))

                img_path = (card.get("image") or "").strip()
                if img_path and Path(img_path).exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("Figure not found. Check the `image` path in `precomputed_cards.json`.")

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="margin-top:0.8rem;font-size:0.78rem;color:#9ca3af;">
            These views are generated offline from the sample dataset and RAG corpus.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ========== Tab 2: Chat ==========

with tab_chat:
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pending_q = st.session_state.pop("pending_query", None)

    user_q = st.chat_input("Ask a question about the patient or ECMO guidance…")

    submitted = False
    if pending_q:
        user_q = pending_q
        submitted = True
    elif user_q:
        submitted = True

    if submitted and rag is not None:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("Thinking…"):
            hits = rag.search(user_q, topk=topk) if rag else []
            prompt = build_prompt_no_citations(
                user_q,
                st.session_state.get("patient_summary", "(no patient uploaded)"),
                hits,
            )
            ans = (
                ollama_generate(ollama_model, prompt, base_url=ollama_base_url)
                or "LLM unavailable. Please confirm the Ollama server."
            )

        st.session_state["messages"].append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)

    if st.button("Clear chat"):
        st.session_state["messages"] = []

    st.markdown("##### 🔎 Example Questions")
    st.caption("Clicking an example will automatically send it into the chat.")
    examples = [
        "Give a concise summary of the last 30 minutes and the next 2 actions.",
        "Explain the spikes in HR over the 7 days.",
        "Which vitals contribute most to the current risk and why?",
        "How can we reduce risk if O2 sat is low and SBP is borderline?",
        "Write a one-paragraph note for the attending based on current status.",
        "Give some advice on the infant patient?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(examples):
        if cols[i % 3].button(q, use_container_width=True, key=f"ex_{i}"):
            st.session_state["pending_query"] = q
            st.rerun()   # <-- updated from experimental_rerun


