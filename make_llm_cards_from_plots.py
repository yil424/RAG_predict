# -*- coding: utf-8 -*-
"""
Use local model + existing plots/data to generate
curated Q&A cards for your second tab.

Outputs:
    precomputed_cards.json

Requires:
    - sim_ecmo_timeseries.csv
    - sim_patient_hazard_scores.csv
    - hazard_model.pkl
    - hazard_model_features.json
    - predict_risk_baseline_3.py  (for build_features_single)
    - shap (pip install shap)
    - local LLM endpoint (default: http://localhost:11434, model llama3.1:8b)

Already generated:
    - fig_gauge.png
    - fig_next6h_danger.png
    - fig_shap_violin.png
    - fig_ridgeline.png
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
import joblib
import shap

from predict_risk_baseline_3 import build_features_single  # uses your updated version


# ---------- Config ----------

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

HAZARD_CSV = "sim_patient_hazard_scores.csv"
PATIENT_CSV = "sim_ecmo_timeseries.csv"
MODEL_PATH = "hazard_model.pkl"
FEAT_PATH = "hazard_model_features.json"

OUT_JSON = "precomputed_cards.json"


# ---------- Helpers ----------

def ollama_generate(prompt: str,
                    model: str = OLLAMA_MODEL,
                    base_url: str = OLLAMA_BASE,
                    timeout: int = 120) -> str:
    """Simple non-streaming call to local Ollama."""
    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def _np(x):
    return np.asarray(x, float)


def predict_p6h_from_hazard(
    hazard_csv: str = HAZARD_CSV,
    step_sec: int = 15,
    calib_path: str = "out/calib_6h_lr.json",
) -> float:
    """Very close to your app_chat_pretty logic."""
    df = pd.read_csv(hazard_csv)
    if "hazard_prob" not in df.columns:
        return 0.02

    s = _np(df["hazard_prob"])
    if s.size < 8:
        return float(np.clip(s[-1] if s.size else 0.02, 0.001, 0.99))

    # EWMA-esque smooth
    alpha = 0.25
    sm = np.zeros_like(s)
    acc = 0.0
    for i, v in enumerate(s):
        acc = alpha * v + (1 - alpha) * acc
        sm[i] = acc
    r_now = float(np.clip(sm[-1], 0.0, 1.0))

    # last 1h window
    w1h = max(2, int(round(3600 / step_sec)))
    seg = sm[-w1h:] if sm.size >= w1h else sm
    r_mean = float(np.clip(np.nanmean(seg), 0.0, 1.0))
    r_max = float(np.clip(np.nanmax(seg), 0.0, 1.0))

    # simple slope (30min)
    k = max(2, int(round(1800 / step_sec)))
    sub = sm[-k:] if sm.size >= k else sm
    x = np.arange(len(sub), dtype=float)
    x = x - x.mean()
    denom = (x * x).sum() or 1.0
    sl = float(((x * (sub - sub.mean())).sum()) / denom)

    # logistic calibration (optionally load)
    w = np.array([2.2, 1.4, 1.0, 0.8], float)
    b = -2.0
    try:
        p = Path(calib_path)
        if p.exists():
            blob = json.loads(p.read_text(encoding="utf-8"))
            if "w" in blob and "b" in blob:
                w = _np(blob["w"])
                b = float(blob["b"])
    except Exception:
        pass

    z = float(b + np.dot(w, np.array([r_now, r_mean, r_max, max(0.0, sl)], float)))
    p6 = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(p6, 0.001, 0.99))


# ---------- Card builders ----------

def card_next6h_gauge() -> Dict:
    p6 = predict_p6h_from_hazard()
    pct = round(p6 * 100, 1)

    question = (
        "What is the predicted risk of a serious decompensation event "
        "in the next 6 hours for this patient, and how should we interpret it clinically?"
    )

    prompt = f"""
You are an ECMO / single-ventricle infant intensivist.
You are given a pre-computed risk gauge for a single patient.

Next 6h event probability (serious decompensation or escalation): {pct:.1f}%.

In 4-6 short bullet points, explain:
- Whether this probability is low / moderate / high in this context
- How a bedside team should respond (e.g. reassurance vs closer surveillance)
- One actionable recommendation on documentation or communication

Be concise, clinically realistic, and avoid hedging language like "as an AI".
Answer in English.
"""

    answer = ollama_generate(prompt)
    return {
        "id": "next6h_gauge",
        "question": question,
        "answer": answer,
        "image": "fig_gauge.png",
    }


def card_next6h_curve() -> Dict:
    """Card for the smooth next-6h danger curve."""
    df = pd.read_csv(HAZARD_CSV)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    p = _np(df["hazard_prob"])
    # last 24h summary
    tail = df.iloc[-int(24 * 3600 / 15):] if len(df) > (24 * 3600 / 15) else df
    p_tail = _np(tail["hazard_prob"])
    mean_tail = float(p_tail.mean())
    max_tail = float(p_tail.max())
    last = float(p_tail[-1])

    question = (
        "How should we interpret the recent hazard and the predicted risk "
        "over the next 6 hours for this patient?"
    )

    prompt = f"""
You are reviewing a line plot titled:
"Next 6h danger (square root risk)" for a single-ventricle infant on ECMO.

Context from the last ~24h:
- Mean model hazard probability: {mean_tail:.3f}
- Peak hazard probability: {max_tail:.3f}
- Latest hazard probability: {last:.3f}

The figure shows:
- A smoothed hazard curve over the past 24h (square-root scaled to visually amplify small changes).
- A subtle, bounded 6h forecast band and dashed forecast curve.

In 1 short paragraph plus 3 bullets, explain:
- Overall risk trend (stable / rising / falling)
- How to interpret the 6h forecast qualitatively (e.g. "no abrupt surge expected")
- One or two concrete monitoring / readiness actions

Do NOT restate numeric values mechanically; summarize them.
Answer in English.
"""

    answer = ollama_generate(prompt)
    return {
        "id": "next6h_curve",
        "question": question,
        "answer": answer,
        "image": "fig_next6h_danger.png",
    }


def card_shap_violin() -> Dict:
    if not Path(MODEL_PATH).exists() or not Path(FEAT_PATH).exists():
        raise FileNotFoundError("Need hazard_model.pkl and hazard_model_features.json for SHAP card.")

    model = joblib.load(MODEL_PATH)
    feat_cols = json.loads(Path(FEAT_PATH).read_text(encoding="utf-8"))

    df = pd.read_csv(PATIENT_CSV)
    feats, _, _ = build_features_single(
        df=df,
        time_col="AR",
        freq_sec=15,
        lead_min=15,
        cooldown_min=30,
    )

    # align
    X = feats.copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].fillna(0.0)

    # window
    win = X.iloc[-400:] if len(X) > 400 else X

    # SHAP
    explainer = shap.Explainer(model, win.sample(min(800, len(win)), random_state=0))
    sv = explainer(win, check_additivity=False)
    vals = sv.values
    if vals.ndim == 3:   # [n, feat, class]
        vals = vals[:, :, 1]
    mean_abs = np.mean(np.abs(vals), axis=0)
    order = np.argsort(-mean_abs)[:12]

    top = []
    for idx in order:
        fname = win.columns[idx]
        score = float(mean_abs[idx])
        # simple pretty naming
        parts = fname.split("__")
        if len(parts) == 3:
            base, winlab, stat = parts
            stat_map = {
                "min": "min", "max": "max", "mean": "mean",
                "last": "last", "delta": "change", "slope": "trend", "std": "var"
            }
            label = f"{base} {stat_map.get(stat, stat)} ({winlab})"
        else:
            label = fname
        top.append(f"- {label}: {score:.3f}")

    top_txt = "\n".join(top)

    question = (
        "Which variables are most responsible for the current hazard "
    )

    prompt = f"""
You are reviewing a SHAP violin plot titled:
"Top features influencing predicted hazard"
for a hazard model in a single-ventricle infant on ECMO.

The SHAP ranking for this patient (top ~12 features by mean |SHAP|) is:

{top_txt}

Using this information, in 4-6 bullet points:
- Summarize which physiological signals are most protective vs most concerning
- Translate them into plain clinical language (SpO2, HR, NIRS, blood pressure, etc.)
- Avoid internal feature-code jargon; talk like a bedside intensivist.
- End with one sentence on how this supports the overall "risk is currently low/moderate" message.

Answer in English.
"""

    answer = ollama_generate(prompt)
    return {
        "id": "shap_violin",
        "question": question,
        "answer": answer,
        "image": "fig_shap_violin.png",
    }


def card_ridgeline() -> Dict:
    df = pd.read_csv(PATIENT_CSV)
    if "AR" in df.columns:
        df["AR"] = pd.to_datetime(df["AR"])
        df = df.set_index("AR").sort_index()

    vitals = []
    for v in ["O2_sat", "HR", "SBP", "RR", "NIRS", "DBP"]:
        if v in df.columns:
            by_day = df[v].groupby(pd.Grouper(freq="D"))
            daily_med = by_day.median()
            if daily_med.notna().sum() >= 2:
                start = daily_med.first_valid_index()
                end = daily_med.last_valid_index()
                vitals.append(
                    f"- {v}: typical range {daily_med.min():.1f}–{daily_med.max():.1f} "
                    f"from {start.date() if start else ''} to {end.date() if end else ''}"
                )

    vitals_txt = "\n".join(vitals) if vitals else "- (insufficient data)"

    question = (
        "What is the distribution of various vitals for the infants over the past few days"
    )

    prompt = f"""
You are reviewing ridgeline distribution plots for multiple days of key vitals
(SpO2, HR, SBP, RR, NIRS, DBP) in a single-ventricle infant.

Data-derived summary (rough, from medians by day):
{vitals_txt}

In 1 short paragraph plus 3 bullets:
- Comment on stability vs drift for oxygenation, blood pressure, and NIRS
- Highlight any day-to-day shifts that could signal improving or worsening balance
- State clearly whether there is evidence for progressive deterioration

Be concise and clinically focused. Answer in English.
"""

    answer = ollama_generate(prompt)
    return {
        "id": "ridgeline",
        "question": question,
        "answer": answer,
        "image": "fig_ridgeline.png",
    }


# ---------- Main ----------

def main():
    cards: List[Dict] = []
    print("[*] Building card: gauge (next6h probability)")
    cards.append(card_next6h_gauge())

    print("[*] Building card: next6h hazard curve")
    cards.append(card_next6h_curve())

    print("[*] Building card: SHAP violin")
    cards.append(card_shap_violin())

    print("[*] Building card: ridgeline")
    cards.append(card_ridgeline())

    Path(OUT_JSON).write_text(
        json.dumps(cards, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Saved {OUT_JSON} with {len(cards)} cards.")


if __name__ == "__main__":
    main()
