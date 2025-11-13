# -*- coding: utf-8 -*-
"""
predict_risk_baseline_3.py

cohort train:
  python predict_risk_baseline_3.py train  --cohort-dir cohort  --pattern "patient_*.csv"  --time-col AR  --freq-sec 15  --lead-min 15  --hazard-win-min 5  --cooldown-min 30  --model hgb

sim test:
  python predict_risk_baseline_3.py apply --csv sim_ecmo_timeseries.csv --time-col AR --freq-sec 15 --lead-min 15 --hazard-win-min 5 --cooldown-min 30 --alarm-opt --lambda1 2.5 --lambda2 1 --lambda3 0.4 --recall-floor 0.03 --recall-penalty 2.5 --alpha-grid "0.2,0.4"
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
import joblib

# --------------------------- CLI ---------------------------

def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # TRAIN
    ap_train = sub.add_parser("train")
    ap_train.add_argument("--cohort-dir", default="cohort")
    ap_train.add_argument("--pattern", default="patient_*.csv")
    ap_train.add_argument("--time-col", default="AR")
    ap_train.add_argument("--freq-sec", type=int, default=15)
    ap_train.add_argument("--lead-min", type=int, default=15,
                          help="lead time before onset for hazard label")
    ap_train.add_argument("--hazard-win-min", type=int, default=5,
                          help="positive window length (minutes) BEFORE j-L for hazard label")
    ap_train.add_argument("--hazard-delta-min", type=int, default=5)
    ap_train.add_argument("--cooldown-min", type=int, default=30)
    ap_train.add_argument("--model", choices=["hgb", "rf", "logreg"], default="hgb")
    ap_train.add_argument("--max-hist-min", type=int, default=60)
    ap_train.add_argument("--knowledge-json", default="knowledge_rules.json")

    # APPLY
    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--csv", default="sim_ecmo_timeseries.csv")
    ap_apply.add_argument("--time-col", default="AR")
    ap_apply.add_argument("--freq-sec", type=int, default=15)
    ap_apply.add_argument("--lead-min", type=int, default=15,
                          help="minimum lead to count a TP (硬约束)")
    ap_apply.add_argument("--hazard-win-min", type=int, default=5,
                          help="positive window length (minutes) BEFORE j-L for hazard label")
    ap_apply.add_argument("--hazard-delta-min", type=int, default=5)
    ap_apply.add_argument("--cooldown-min", type=int, default=30)
    ap_apply.add_argument("--knowledge-json", default="knowledge_rules.json")
    ap_apply.add_argument("--alarm-opt", action="store_true",
                          help="run alarm policy search on this patient using hazard probs")

    # 论文损失的权重 + 其它搜索超参
    ap_apply.add_argument("--lambda1", type=float, default=1.0, help="weight for (1-Precision)")
    ap_apply.add_argument("--lambda2", type=float, default=1.0, help="weight for (1-Recall)")
    ap_apply.add_argument("--lambda3", type=float, default=0.5, help="weight for normalized median lead")
    ap_apply.add_argument("--recall-floor", type=float, default=0.01,
                          help="soft floor on recall; below this add penalty")
    ap_apply.add_argument("--recall-penalty", type=float, default=1.0,
                          help="penalty strength when recall < floor")
    ap_apply.add_argument("--alpha-grid", type=str, default="0.2,0.4,0.6",
                          help="EWMA alphas to try, comma-separated in (0,1]")

    return ap.parse_args()

# --------------------------- CONSTANTS ---------------------------

COLUMN_ALIASES = {
    "SpO2": "O2_sat",
    "O2 sats": "O2_sat",
    "O2_sats": "O2_sat",
    "PaO2": "PAO2",
    "PaO₂": "PAO2",
    "CVP": "CVRP",
    "Central Venous/Right atrial pressure": "CVRP",
    "Base excess/Base Deficit": "BE",
    "Lactates": "Lactate",
    "Mixed Venous 02 Sats": "MVO2_sats",
    "Mixed Venous O2 Sats": "MVO2_sats",
    "SvO2": "MVO2_sats",
    "Creatinine": "CR",
    "BUN mg/dl": "BUN",
}

ALL_VITALS = ["HR", "RR", "O2_sat", "NIRS", "SBP", "DBP", "CVRP"]
LABS_6H = ["pH", "BE", "PAO2", "Lactate", "MVO2_sats"]
LABS_24H = ["CR", "BUN"]

THRESHOLDS_FALLBACK = {
    "HR_low": 90, "HR_high": 190,
    "RR_low": 42, "RR_high": 62,
    "O2_sat_low": 72,
    "NIRS_low": 58,
    "PAO2_low": 45,
    "SBP_low": 58, "SBP_high": 90,
    "DBP_low": 32, "DBP_high": 60,
    "CVRP_low": 3, "CVRP_high": 11.5,
    "pH_low": 7.25, "pH_high": 7.55,
    "BE_low": -5, "BE_high": 8,
    "Lactate_high": 4.0,
    "MVO2_sats_low": 55,
}

# --------------------------- UTILS ---------------------------

def minutes_to_steps(m: int, freq_sec: int) -> int:
    return int((m * 60) / freq_sec)

def minutes_to_steps_safe(m: int, freq_sec: int) -> int:
    return max(1, int(round((m * 60) / freq_sec)))

def rolling_last_np(a: np.ndarray) -> float:
    for x in a[::-1]:
        if np.isfinite(x):
            return float(x)
    return np.nan

def rolling_delta_np(a: np.ndarray) -> float:
    first = np.nan
    for x in a:
        if np.isfinite(x):
            first = x
            break
    last = rolling_last_np(a)
    return np.nan if (np.isnan(first) or np.isnan(last)) else float(last - first)

def rolling_slope_per_min_np(a: np.ndarray, freq_sec: int) -> float:
    idx = np.arange(len(a), dtype=float) * (freq_sec / 60.0)
    mask = np.isfinite(a)
    if mask.sum() < 2:
        return np.nan
    x = idx[mask]
    y = a[mask]
    x = x - x.mean()
    denom = (x ** 2).sum()
    if denom <= 1e-9:
        return 0.0
    beta = (x * y).sum() / denom
    return float(beta)

# --------------------------- EWMA SMOOTH ---------------------------

def ewma_smooth(probs: np.ndarray, alpha: float) -> np.ndarray:
    """s_t = alpha * p_t + (1-alpha) * s_{t-1}"""
    s = np.empty_like(probs, dtype=float)
    if len(probs) == 0:
        return probs.astype(float)
    s[0] = float(probs[0])
    a = float(alpha)
    b = 1.0 - a
    for t in range(1, len(probs)):
        s[t] = a * float(probs[t]) + b * s[t-1]
    return s

# --------------------------- KNOWLEDGE RULES ---------------------------

def load_knowledge_rules(path: str) -> Dict[str, Dict]:
    p = Path(path)
    rules: Dict[str, Dict] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                thr = v.get("suggested_threshold")
                direction = v.get("direction")
                if thr is None or direction not in ("low", "high"):
                    continue
                rules[k] = {
                    "threshold": float(thr),
                    "direction": direction,
                    "source": "rag",
                }
            if rules:
                print(f"[kg] Loaded {len(rules)} rules from {path}")
                return rules
        except Exception as e:
            print(f"[kg] Failed to load {path}, fallback to static thresholds. Error: {e}")

    for name, thr in THRESHOLDS_FALLBACK.items():
        if name.endswith("_low"):
            rules[name] = {"threshold": float(thr), "direction": "low", "source": "fallback"}
        elif name.endswith("_high"):
            rules[name] = {"threshold": float(thr), "direction": "high", "source": "fallback"}
    print(f"[kg] Using fallback thresholds, n={len(rules)}")
    return rules

GLOBAL_KG_RULES = load_knowledge_rules("knowledge_rules.json")

def compute_knowledge_features(df: pd.DataFrame,
                               freq_sec: int,
                               rules: Dict[str, Dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    flag_cols: List[str] = []
    for name, cfg in rules.items():
        thr = cfg.get("threshold")
        direction = cfg.get("direction")
        if thr is None or direction not in ("low", "high"):
            continue
        var = None
        if name.endswith("_low"):
            var = name[:-4]
        elif name.endswith("_high"):
            var = name[:-5]
        if not var or var not in df.columns:
            continue
        s = pd.to_numeric(df[var], errors="coerce")
        col_flag = f"kg_{name}_flag"
        out[col_flag] = (s < thr).astype(int) if direction == "low" else (s > thr).astype(int)
        flag_cols.append(col_flag)

    for col in flag_cols:
        for wmin in [15, 60]:
            w = minutes_to_steps(wmin, freq_sec)
            if w > 1:
                out[f"{col}__{wmin}m_frac"] = out[col].rolling(
                    w, min_periods=max(2, w // 3)).mean()

    if flag_cols:
        rf_sum = out[flag_cols].sum(axis=1)
        out["kg_multi_redflags"] = rf_sum
        out["kg_multi_any2plus"] = (rf_sum >= 2).astype(int)
    return out

# --------------------------- HAZARD LABEL ---------------------------

def find_event_onsets(y_now: np.ndarray,
                      cooldown_min: int,
                      freq_sec: int) -> List[int]:
    y = y_now.astype(int).copy()
    n = len(y)
    cd = minutes_to_steps(cooldown_min, freq_sec)
    onsets: List[int] = []
    i = 0
    while i < n:
        if y[i] == 1:
            onsets.append(i)
            j = i + 1
            while j < n and y[j] == 1:
                j += 1
            i = j + cd
        else:
            i += 1
    return onsets

def make_hazard_label_window(y_now: np.ndarray,
                             lead_min: int,
                             win_min: int,
                             cooldown_min: int,
                             freq_sec: int) -> np.ndarray:
    n = len(y_now)
    h = np.zeros(n, dtype=int)
    onsets = find_event_onsets(y_now, cooldown_min, freq_sec)
    if not onsets:
        return h
    L = minutes_to_steps(lead_min, freq_sec)
    W = max(1, minutes_to_steps(win_min, freq_sec))
    for j in onsets:
        t2 = j - L
        t1 = max(0, t2 - W + 1)
        if t1 <= t2:
            h[t1:t2+1] = 1
    return h

# --------------------------- FEATURE BUILDING ---------------------------

def build_features_single(df: pd.DataFrame,
                          time_col: str,
                          freq_sec: int,
                          lead_min: int,
                          hazard_win_min: int,
                          cooldown_min: int,
                          kg_rules: Optional[Dict[str, Dict]] = None
                          ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    if kg_rules is None:
        kg_rules = GLOBAL_KG_RULES
    df = df.copy()
    df.rename(columns=COLUMN_ALIASES, inplace=True)
    if time_col not in df.columns:
        raise ValueError(f"TIME_COL '{time_col}' not in columns.")
    if "label" not in df.columns:
        raise ValueError("Need 'label' column (0/1).")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    vitals = [c for c in ALL_VITALS if c in df.columns]
    if vitals:
        df[vitals] = df[vitals].apply(pd.to_numeric, errors="coerce")
        df[vitals] = df[vitals].ffill(limit=minutes_to_steps(60, freq_sec))

    t_minutes = (df.index - df.index[0]).total_seconds() / 60.0
    t_minutes = pd.Series(t_minutes, index=df.index)

    L = minutes_to_steps(lead_min, freq_sec)
    vitals_past = df[vitals].shift(L) if vitals else pd.DataFrame(index=df.index)

    derived_cols: List[str] = []
    if ("HR" in df.columns) and ("SBP" in df.columns):
        si = vitals_past["HR"] / vitals_past["SBP"].replace(0, np.nan)
        vitals_past = vitals_past.assign(shock_index=si)
        derived_cols.append("shock_index")
    if ("SBP" in df.columns) and ("DBP" in df.columns):
        pp = vitals_past["SBP"] - vitals_past["DBP"]
        vitals_past = vitals_past.assign(pulse_pressure=pp)
        derived_cols.append("pulse_pressure")

    vitals_all = [c for c in vitals_past.columns if c in vitals] + [
        c for c in derived_cols if c in vitals_past.columns
    ]

    feat_blocks: List[pd.DataFrame] = []

    # Labs
    for lab in (LABS_6H + LABS_24H):
        if lab not in df.columns:
            continue
        s = pd.to_numeric(df[lab], errors="coerce")
        s_past = s.shift(L)
        last = s_past.ffill()
        last_obs_time = t_minutes.where(s_past.notna()).ffill()
        mins_since = (t_minutes - last_obs_time).astype(float)

        blk = pd.DataFrame(index=df.index)
        blk[f"{lab}__last"] = last
        blk[f"{lab}__mins_since_last"] = mins_since
        blk[f"{lab}__recent6h"] = (mins_since <= 60 * 6).astype(int)
        blk[f"{lab}__recent24h"] = (mins_since <= 60 * 24).astype(int)
        feat_blocks.append(blk)

    # Rolling vitals
    for wmin in [5, 15, 60]:
        w = minutes_to_steps(wmin, freq_sec)
        if w <= 1 or not vitals_all:
            continue
        r = vitals_past[vitals_all].rolling(w, min_periods=max(2, w // 3))
        blk = pd.DataFrame(index=df.index)
        for c in vitals_all:
            arr_min = r[c].min()
            arr_max = r[c].max()
            arr_mean = r[c].mean()
            arr_std = r[c].std()
            arr_last = r[c].apply(lambda a: rolling_last_np(a.to_numpy()), raw=False)
            arr_delta = r[c].apply(lambda a: rolling_delta_np(a.to_numpy()), raw=False)
            arr_slope = r[c].apply(lambda a: rolling_slope_per_min_np(a.to_numpy(), freq_sec), raw=False)
            blk[f"{c}__{wmin}m__min"] = arr_min
            blk[f"{c}__{wmin}m__max"] = arr_max
            blk[f"{c}__{wmin}m__mean"] = arr_mean
            blk[f"{c}__{wmin}m__std"] = arr_std
            blk[f"{c}__{wmin}m__last"] = arr_last
            blk[f"{c}__{wmin}m__delta"] = arr_delta
            blk[f"{c}__{wmin}m__slope"] = arr_slope
        feat_blocks.append(blk)

    # Knowledge-guided
    kg_feats = compute_knowledge_features(df, freq_sec=freq_sec, rules=kg_rules)
    feat_blocks.append(kg_feats)

    feats = pd.concat(feat_blocks, axis=1)

    y_now = df["label"].fillna(0).astype(int).to_numpy()
    y_hazard_np = make_hazard_label_window(
        y_now=y_now,
        lead_min=lead_min,
        win_min=hazard_win_min,
        cooldown_min=cooldown_min,
        freq_sec=freq_sec,
    )
    y_hazard = pd.Series(y_hazard_np, index=df.index, name="hazard")

    keep_idx = feats.dropna().index.intersection(y_hazard.index)
    feats = feats.loc[keep_idx]
    y_hazard = y_hazard.loc[keep_idx]
    y_now_aligned = pd.Series(y_now, index=df.index).loc[keep_idx].to_numpy()

    print(f"[build_single] feats={feats.shape}, hazard_pos_rate={y_hazard.mean():.4f}")
    return feats, y_hazard, y_now_aligned

def build_features_cohort(cohort_dir: str,
                          pattern: str,
                          time_col: str,
                          freq_sec: int,
                          lead_min: int,
                          hazard_win_min: int,
                          cooldown_min: int,
                          kg_rules: Optional[Dict[str, Dict]] = None
                          ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    if kg_rules is None:
        kg_rules = GLOBAL_KG_RULES
    cohort_path = Path(cohort_dir)
    csvs = sorted(cohort_path.glob(pattern))
    assert csvs, f"No CSV matched {pattern} in {cohort_dir}"

    X_list, y_list, pid_list = [], [], []
    for p in csvs:
        pid = p.stem
        dfp = pd.read_csv(p)
        Xi, yi, _ = build_features_single(
            df=dfp,
            time_col=time_col,
            freq_sec=freq_sec,
            lead_min=lead_min,
            hazard_win_min=hazard_win_min,
            cooldown_min=cooldown_min,
            kg_rules=kg_rules,
        )
        Xi = Xi.copy()
        Xi["patient_id"] = pid
        X_list.append(Xi)
        y_list.append(yi)
        pid_list.append(pd.Series(pid, index=Xi.index, name="patient_id"))
        print(f"[cohort] {pid}: feats={Xi.shape}, hazard_pos_rate={yi.mean():.4f}")

    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)
    pids = pd.concat(pid_list, axis=0)

    common = X_all.index.intersection(y_all.index)
    X_all = X_all.loc[common]
    y_all = y_all.loc[common]
    pids = pids.loc[common]
    return X_all, y_all, pids

# --------------------------- SPLIT & MODEL ---------------------------

def patient_group_split(X: pd.DataFrame, y: pd.Series, pids: pd.Series,
                        train_size=0.7, val_size=0.15, test_size=0.15,
                        random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    tr_idx, rem_idx = next(gss1.split(X, y, groups=pids))
    X_rem = X.iloc[rem_idx]; y_rem = y.iloc[rem_idx]; pid_rem = pids.iloc[rem_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size/(val_size+test_size),
                             random_state=random_state+1)
    va_rel, te_rel = next(gss2.split(X_rem, y_rem, groups=pid_rem))
    va_idx = rem_idx[va_rel]; te_idx = rem_idx[te_rel]
    return tr_idx, va_idx, te_idx

def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("lr", LogisticRegression(max_iter=400, class_weight="balanced"))
        ])
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    else:
        return HistGradientBoostingClassifier(
            learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=0.1, random_state=42)

def predict_proba_or_score(clf, X: pd.DataFrame) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    else:
        return clf.decision_function(X)

# --------------------------- ALARM POLICY ---------------------------

def simulate_alarms(time_index: pd.DatetimeIndex,
                    sprob: np.ndarray,
                    tau: float,
                    persist: int,
                    refractory_min: int,
                    freq_sec: int) -> List[int]:
    n = len(sprob)
    alarms: List[int] = []
    refractory_steps = max(minutes_to_steps_safe(refractory_min, freq_sec),
                           minutes_to_steps_safe(1, freq_sec))
    consec = 0
    last_alarm = -10**9
    for t in range(n):
        if sprob[t] >= tau:
            consec += 1
        else:
            consec = 0
        if consec >= persist and (t - last_alarm) >= refractory_steps:
            a = t - persist + 1
            alarms.append(a)
            last_alarm = a
            consec = 0
    return alarms

def alarm_metrics(alarms: List[int],
                  y_now: np.ndarray,
                  time_index: pd.DatetimeIndex,
                  horizon_min: int = 60,
                  min_lead_required_min: int = 0) -> Dict[str, float]:
    y_now = np.asarray(y_now, dtype=int).ravel()
    n = len(y_now)
    assert len(time_index) == n

    onsets: List[int] = []
    prev = 0
    for i, v in enumerate(y_now):
        if v == 1 and prev == 0:
            onsets.append(i)
        prev = v

    if len(alarms) == 0 and len(onsets) == 0:
        return {
            "precision": float("nan"),
            "recall": 0.0,
            "median_TTD_min": float("nan"),
            "tp": 0, "fp": 0, "fn": 0,
            "n_alarms": 0, "n_onsets": 0,
            "horizon_min": int(horizon_min),
        }

    horizon = pd.Timedelta(minutes=int(horizon_min))
    min_lead_req = float(min_lead_required_min)

    onset_used = [False] * len(onsets)
    alarm_used = [False] * len(alarms)
    ttd_list: List[float] = []

    for j, onset_idx in enumerate(onsets):
        onset_t = time_index[onset_idx]
        best_i, best_delta = None, None
        for i, a_idx in enumerate(alarms):
            if alarm_used[i]:
                continue
            at = time_index[a_idx]
            if at <= onset_t and (onset_t - at) <= horizon:
                delta_min = (onset_t - at).total_seconds() / 60.0
                if delta_min + 1e-9 < min_lead_req:
                    continue
                if best_delta is None or delta_min < best_delta:
                    best_delta = delta_min
                    best_i = i
        if best_i is not None:
            onset_used[j] = True
            alarm_used[best_i] = True
            ttd_list.append(best_delta)

    tp = int(sum(onset_used))
    fp = int(sum(1 for u in alarm_used if not u))
    fn = int(len(onsets) - tp)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    median_ttd = float(np.median(ttd_list)) if ttd_list else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "median_TTD_min": median_ttd,
        "tp": tp, "fp": fp, "fn": fn,
        "n_alarms": int(len(alarms)),
        "n_onsets": int(len(onsets)),
        "horizon_min": int(horizon_min),
    }

def search_best_alarm_policy(time_index: pd.DatetimeIndex,
                             probs: np.ndarray,
                             y_now: np.ndarray,
                             freq_sec: int,
                             horizons_min: Tuple[int, ...] = (60, 180, 360),
                             min_lead_required_min: int = 0,
                             lambda1: float = 1.0,
                             lambda2: float = 1.0,
                             lambda3: float = 0.5,
                             recall_floor: float = 0.01,
                             recall_penalty: float = 1.0,
                             alphas: Tuple[float, ...] = (0.2, 0.4, 0.6),
                             min_avg_recall_required: float = 0.05   # 平均召回硬要求
                             ) -> None:
    persists = [1, 2, 3, 4]
    refrs = [5, 15, 30, 60]

    best = None; best_L = +1e9
    fallback_best = None; fallback_best_recall = -1.0; fallback_best_L = +1e9

    for alpha in alphas:
        sprob = ewma_smooth(probs, float(alpha))

        qs = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
        tau_q = np.quantile(sprob, qs, method="linear")
        tau_fixed = np.array([0.01, 0.02, 0.03, 0.05], dtype=float)
        taus = np.unique(np.concatenate([tau_q, tau_fixed]))
        taus = taus[(taus >= 0.0) & (taus <= 1.0)]
        if len(taus) == 0:
            taus = np.array([0.05], dtype=float)

        for tau in taus:
            for p in persists:
                for r in refrs:
                    alarms = simulate_alarms(time_index, sprob, float(tau), p, r, freq_sec)

                    Ms = []
                    for H in horizons_min:
                        m = alarm_metrics(
                            alarms, y_now, time_index,
                            horizon_min=int(H),
                            min_lead_required_min=int(min_lead_required_min),
                        )
                        Ms.append(m)

                    Ps=[]; Rs=[]; Ls=[]
                    for H, m in zip(horizons_min, Ms):
                        P = m["precision"]; R = m["recall"]; Lead_med = m["median_TTD_min"]
                        if not np.isfinite(P): P = 0.0
                        if not np.isfinite(R): R = 0.0
                        if not np.isfinite(Lead_med): Lead_med = 0.0
                        Ps.append(P); Rs.append(R)
                        lead_norm = float(np.clip(Lead_med / float(H), 0.0, 1.0))
                        Lh = lambda1*(1.0 - P) + lambda2*(1.0 - R) - lambda3*lead_norm
                        if R < recall_floor:
                            Lh += recall_penalty * (recall_floor - R)
                        Ls.append(Lh)

                    avgP = float(np.mean(Ps)) if Ps else 0.0
                    avgR = float(np.mean(Rs)) if Rs else 0.0
                    L = float(np.mean(Ls)) if Ls else +1e9

                    if (avgR > fallback_best_recall) or (np.isclose(avgR, fallback_best_recall) and L < fallback_best_L):
                        fallback_best = {
                            "alpha": float(alpha), "tau": float(tau),
                            "persist": int(p), "refractory_min": int(r),
                            "loss": L,
                            "metrics": {int(H): Ms[i] for i, H in enumerate(horizons_min)},
                            "avg_precision": avgP, "avg_recall": avgR,
                            "avg_median_lead": float(np.nanmean([x["median_TTD_min"] if np.isfinite(x["median_TTD_min"]) else 0.0 for x in Ms])),
                        }
                        fallback_best_recall = avgR; fallback_best_L = L

                    # 平均召回硬约束
                    if avgR < float(min_avg_recall_required):
                        continue

                    if L < best_L:
                        best_L = L
                        best = {
                            "alpha": float(alpha), "tau": float(tau),
                            "persist": int(p), "refractory_min": int(r),
                            "loss": L,
                            "metrics": {int(H): Ms[i] for i, H in enumerate(horizons_min)},
                            "avg_precision": avgP, "avg_recall": avgR,
                            "avg_median_lead": float(np.nanmean([x["median_TTD_min"] if np.isfinite(x["median_TTD_min"]) else 0.0 for x in Ms])),
                        }

    chosen = best if best is not None else fallback_best
    if chosen is None:
        print("[alarm] No valid alarm policy found."); return
    if best is None:
        print(f"[alarm] No candidate met avg recall ≥ {min_avg_recall_required:.3f}. Falling back to the highest-recall candidate.")

    print("[alarm] Best policy on this patient:")
    print(f"  alpha={chosen['alpha']:.2f}, threshold={chosen['tau']:.4f}, "
          f"persistence={chosen['persist']} steps, refractory={chosen['refractory_min']} min"
          f"  | lead_min={int(min_lead_required_min)} min, "
          f"L={chosen['loss']:.3f}, avg(P)={chosen['avg_precision']:.3f}, "
          f"avg(R)={chosen['avg_recall']:.3f}, avg(medianLead)={chosen['avg_median_lead']:.1f} min")
    for H in sorted(chosen["metrics"].keys()):
        m = chosen["metrics"][H]
        prec = m["precision"]; prec_str = f"{prec:.3f}" if np.isfinite(prec) else "nan"
        med = m["median_TTD_min"] if np.isfinite(m["median_TTD_min"]) else float('nan')
        print(f"  @H={H:>3d} min: precision={prec_str}, recall={m['recall']:.3f}, "
              f"median_lead={med:.1f} min (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, "
              f"alarms={m['n_alarms']}, onsets={m['n_onsets']})")

# --------------------------- MAIN ---------------------------

def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("lr", LogisticRegression(max_iter=400, class_weight="balanced"))
        ])
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    else:
        return HistGradientBoostingClassifier(
            learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=0.1, random_state=42)

def main():
    args = get_args()
    kg_rules = load_knowledge_rules(getattr(args, "knowledge_json", "knowledge_rules.json"))

    if args.mode == "train":
        X_all, y_all, pids = build_features_cohort(
            cohort_dir=args.cohort_dir, pattern=args.pattern,
            time_col=args.time_col, freq_sec=args.freq_sec,
            lead_min=args.lead_min, hazard_win_min=args.hazard_win_min,
            cooldown_min=args.cooldown_min, kg_rules=kg_rules,
        )
        tr, va, te = patient_group_split(X_all, y_all, pids)
        Xtr = X_all.iloc[tr].drop(columns=["patient_id"])
        Xva = X_all.iloc[va].drop(columns=["patient_id"])
        Xte = X_all.iloc[te].drop(columns=["patient_id"])
        ytr, yva, yte = y_all.iloc[tr], y_all.iloc[va], y_all.iloc[te]

        print(f"[split] patients train/val/test = "
              f"{pids.iloc[tr].nunique()}/{pids.iloc[va].nunique()}/{pids.iloc[te].nunique()} "
              f"| hazard_pos = {int(ytr.sum())}/{int(yva.sum())}/{int(yte.sum())}")

        clf = make_model(args.model)

        w_pos = 1.0
        pi = float(ytr.mean())
        if 0.0 < pi < 0.5:
            w_pos = (1.0 - pi) / max(pi, 1e-9) * 0.5
        sw_tr = np.where(ytr.values == 1, w_pos, 1.0)

        clf.fit(Xtr, ytr, sample_weight=sw_tr)

        proba_te = predict_proba_or_score(clf, Xte)
        auroc = roc_auc_score(yte, proba_te) if 0 < yte.sum() < len(yte) else np.nan
        auprc = average_precision_score(yte, proba_te) if yte.sum() > 0 else np.nan
        print(f"[train] Test AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

        proba_va = predict_proba_or_score(clf, Xva)
        proba_tr = predict_proba_or_score(clf, Xtr)
        if yva.sum() > 0:
            p, r, th = precision_recall_curve(yva, proba_va)
            f1 = 2 * p * r / (p + r + 1e-9)
            if len(th) > 0 and np.isfinite(f1).any():
                best = int(np.nanargmax(f1))
                idx = max(0, min(best - 1, len(th) - 1))
                thr = float(th[idx])
            else:
                thr = float(np.quantile(proba_tr, 0.98))
        else:
            thr = float(np.quantile(proba_tr, 0.98))
        thr = float(np.clip(thr, 0.05, 0.95))
        Path("hazard_threshold.json").write_text(
            json.dumps({"hazard_threshold": thr}, indent=2), encoding="utf-8")
        print(f"[train] Learned hazard_threshold={thr:.3f} -> saved to hazard_threshold.json")

        joblib.dump(clf, "hazard_model.pkl")
        feat_cols = list(Xtr.columns)
        Path("hazard_model_features.json").write_text(json.dumps(feat_cols, indent=2),
                                                     encoding="utf-8")
        print("[train] Saved hazard_model.pkl and hazard_model_features.json")

    elif args.mode == "apply":
        if not Path("hazard_model.pkl").exists():
            raise SystemExit("hazard_model.pkl not found. Run train mode first.")
        if not Path("hazard_model_features.json").exists():
            raise SystemExit("hazard_model_features.json not found. Run train mode first.")

        clf = joblib.load("hazard_model.pkl")
        feat_cols = json.loads(Path("hazard_model_features.json").read_text(encoding="utf-8"))

        df = pd.read_csv(args.csv)
        feats, y_hazard, y_now = build_features_single(
            df=df, time_col=args.time_col, freq_sec=args.freq_sec,
            lead_min=args.lead_min, hazard_win_min=args.hazard_win_min,
            cooldown_min=args.cooldown_min, kg_rules=kg_rules,
        )

        X = feats.copy()
        for c in feat_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feat_cols]

        probs = predict_proba_or_score(clf, X)

        out = pd.DataFrame({
            "time": X.index.astype(str),
            "hazard_prob": probs,
            "hazard_label": y_hazard.loc[X.index].values,
            "label_now": y_now[-len(X):],
        })
        out.to_csv("sim_patient_hazard_scores.csv", index=False)
        print("[apply] Saved sim_patient_hazard_scores.csv")

        if args.alarm_opt:
            alphas = tuple(float(x) for x in str(args.alpha_grid).split(",") if x.strip())
            search_best_alarm_policy(
                X.index, probs, y_now, args.freq_sec,
                horizons_min=(60, 180, 360),
                min_lead_required_min=int(args.lead_min),
                lambda1=float(args.lambda1), lambda2=float(args.lambda2), lambda3=float(args.lambda3),
                recall_floor=float(args.recall_floor), recall_penalty=float(args.recall_penalty),
                alphas=alphas,
                min_avg_recall_required=0.05,
            )

if __name__ == "__main__":
    main()





