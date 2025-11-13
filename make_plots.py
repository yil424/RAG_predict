# Two plots：
# 1) Next 6h danger: EWMA hazard (sqrt-scale) + smooth 6h forecast
# 2) SHAP violin: Explaining the main reasons for the recent hazard risk

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import shap

from predict_risk_baseline_3 import build_features_single


def ewma(x, alpha: float = 0.35):
    x = np.asarray(x, float)
    y = np.zeros_like(x, float)
    m = 0.0
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m
        y[i] = m
    return y


def sqrt_clip(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return np.sqrt(x)


def plot_next6h_danger_smooth(
    hazard_csv: str = "sim_patient_hazard_scores.csv",
    out_png: str = "fig_next6h_danger.png",
    alpha: float = 0.35,
    hist_hours: float = 24.0,
    horizon_hours: float = 6.0,
    step_sec: int = 15,
    vis_scale: float = 4.0,
    lookback_min: float = 90.0,
    max_delta_abs: float = 0.15,
    max_slope_per_hr: float = 0.05,
):
    df = pd.read_csv(hazard_csv)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    p_all = df["hazard_prob"].astype(float).values

    ew = ewma(p_all, alpha=alpha)

    sq = sqrt_clip(ew)

    win = max(3, int(round(10 * 60 / step_sec)))
    if len(sq) > win:
        k = np.ones(win, float) / win
        sq_smooth = np.convolve(sq, k, mode="same")
    else:
        sq_smooth = sq.copy()

    end = df["time"].iloc[-1]
    start = end - pd.Timedelta(hours=hist_hours)
    mask_hist = df["time"] >= start

    t_hist = df.loc[mask_hist, "time"]
    y_hist = sq_smooth[mask_hist]

    step = pd.Timedelta(seconds=step_sec)
    horizon_steps = int(round(horizon_hours * 3600 / step_sec))
    t_fore = pd.date_range(end + step, periods=horizon_steps, freq=f"{step_sec}s")

    lb_start = end - pd.Timedelta(minutes=lookback_min)
    mask_lb = df["time"] >= lb_start
    t_lb = df.loc[mask_lb, "time"]
    y_lb = sq_smooth[mask_lb]

    last_val = float(y_hist[-1])

    if len(t_lb) >= 8 and np.nanmax(y_lb) - np.nanmin(y_lb) > 1e-5:
        x_lb = (t_lb - t_lb.iloc[0]).dt.total_seconds().to_numpy()
        coef = np.polyfit(x_lb, y_lb, 1)

        slope = coef[0]  # per second
        max_slope = max_slope_per_hr / 3600.0
        slope = float(np.clip(slope, -max_slope, max_slope))

        x_fore = (t_fore - end).total_seconds().astype(float)
        y_fore_base = last_val + slope * x_fore

        y_fit_lb = last_val + slope * (t_lb - end).dt.total_seconds().to_numpy()
        resid = y_lb - y_fit_lb
        sigma = float(np.nanstd(resid)) if np.isfinite(resid).any() else 0.01
    else:
        y_fore_base = np.full(horizon_steps, last_val, float)
        sigma = 0.01

    y_min = max(0.0, last_val - max_delta_abs)
    y_max = min(1.0, last_val + max_delta_abs)
    y_fore_base = np.clip(y_fore_base, y_min, y_max)

    if horizon_steps > 1:
        period_min = 90.0
        total_min = horizon_hours * 60.0
        x = np.linspace(0.0, 2.0 * np.pi * (total_min / period_min), horizon_steps)
        amp = min(max_delta_abs * 0.35, 2.0 * sigma)
        wiggle = amp * np.sin(x)
        y_fore = y_fore_base + wiggle
    else:
        y_fore = y_fore_base

    y_fore = np.clip(y_fore, y_min, y_max)

    y_low = np.clip(y_fore - 2 * sigma, 0.0, 1.0)
    y_high = np.clip(y_fore + 2 * sigma, 0.0, 1.0)

    y_hist_vis = y_hist * vis_scale
    y_fore_vis = y_fore * vis_scale
    y_low_vis = y_low * vis_scale
    y_high_vis = y_high * vis_scale

    plt.figure(figsize=(14, 3.8))

    plt.plot(
        t_hist,
        y_hist_vis,
        label="Hazard (visual scale)",
        linewidth=2.0,
        color="#2563eb",
    )

    plt.fill_between(
        t_fore,
        y_low_vis,
        y_high_vis,
        color="#bfdbfe",
        alpha=0.5,
        label="6h forecast band",
        linewidth=0,
    )
    plt.plot(
        t_fore,
        y_fore_vis,
        "--",
        color="#1d4ed8",
        linewidth=2.2,
        label="6h forecast",
    )

    plt.title(
        "Next 6h danger (square root risk)",
        fontsize=16,
    )
    plt.ylabel("Risk")
    plt.xlabel("Time")

    ymax = float(
        max(
            0.4,
            np.nanmax(y_hist_vis) * 1.4,
            np.nanmax(y_high_vis) * 1.2,
        )
    )
    plt.ylim(0, ymax)

    plt.grid(alpha=0.18, axis="y")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved {out_png}")


def pretty_feature_name(col: str) -> str:
    base_map = {
        "HR": "Heart rate",
        "RR": "Respiratory rate",
        "O2_sat": "SpO2",
        "NIRS": "Cerebral NIRS",
        "SBP": "Systolic BP",
        "DBP": "Diastolic BP",
        "CVRP": "Central venous pressure",
        "shock_index": "Shock index (HR/SBP)",
        "pulse_pressure": "Pulse pressure (SBP-DBP)",
        "Lactate": "Lactate",
        "PAO2": "PaO₂",
        "MVO2_sats": "Mixed venous O₂ sat",
        "CR": "Creatinine",
        "BUN": "BUN",
    }

    if "__" not in col:
        return base_map.get(col, col)

    base, rest = col.split("__", 1)
    label = base_map.get(base, base)

    time_part = ""
    if rest.startswith(("5m", "15m", "30m", "60m")):
        t = rest.split("__", 1)[0]
        mins = t.replace("m", "")
        time_part = f"past {mins} min"
        rest = rest[len(t) + 2 :] if "__" in rest else ""

    stat_map = {
        "min": "min",
        "max": "max",
        "mean": "mean",
        "std": "variability",
        "last": "last value",
        "delta": "change",
        "slope": "trend",
    }

    stat = ""
    for k, v in stat_map.items():
        if rest.endswith(k):
            stat = v
            break

    pieces = [label]
    if stat:
        pieces.append(stat)
    if time_part:
        pieces.append(f"({time_part})")

    return " ".join(pieces) if len(pieces) > 1 else label


def plot_shap_violin_for_patient(
    patient_csv: str = "sim_ecmo_timeseries.csv",
    time_col: str = "AR",
    freq_sec: int = 15,
    lead_min: int = 15,
    cooldown_min: int = 30,
    model_path: str = "hazard_model.pkl",
    feat_path: str = "hazard_model_features.json",
    out_png: str = "fig_shap_violin.png",
    window_points: int = 400,
    max_display: int = 15,
):

    model = joblib.load(model_path)
    feat_cols = json.loads(Path(feat_path).read_text(encoding="utf-8"))

    df = pd.read_csv(patient_csv)
    feats, y_hazard, _ = build_features_single(
        df=df,
        time_col=time_col,
        freq_sec=freq_sec,
        lead_min=lead_min,
        cooldown_min=cooldown_min,
    )

    X = feats.copy()
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols].fillna(0.0)

    X_win = X if len(X) <= window_points else X.iloc[-window_points:]

    bg_size = min(1000, len(X))
    X_bg = X.sample(bg_size, random_state=0) if len(X) > bg_size else X

    explainer = shap.Explainer(model, X_bg)
    sv = explainer(X_win, check_additivity=False)
    values = sv.values
    if values.ndim == 3:
        values = values[:, :, 1]

    def short_name(col: str) -> str:
        parts = col.split("__")
        if len(parts) == 3:
            base, win, stat = parts
            stat_map = {
                "min": "min", "max": "max", "mean": "mean",
                "last": "last", "delta": "change", "slope": "trend", "std": "var"
            }
            return f"{base} {stat_map.get(stat, stat)} ({win})"
        return col

    pretty_names = [short_name(c) for c in X_win.columns]

    plt.figure(figsize=(7.5, 5.2))
    shap.summary_plot(
        values,
        X_win,
        feature_names=pretty_names,
        plot_type="violin",
        max_display=max_display,
        show=False,
    )
    plt.title(
        "Top features influencing predicted hazard",
        fontsize=14,
    )
    plt.xlabel("Impact on patient hazard")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved {out_png}")


# ========== main ==========

def main():
    plot_next6h_danger_smooth()
    plot_shap_violin_for_patient()


if __name__ == "__main__":
    main()

