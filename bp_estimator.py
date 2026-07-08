"""
Cuffless BP estimation — dual DL path (morphology MLP + waveform CNN) + PWA fusion.

Shen.ai uses proprietary PPG+BP training; we approximate with:
  1. DeepPhys / POS waveform
  2. Pulse-wave morphology MLP
  3. 1D CNN on normalized PPG (BPWaveNet)
  4. Literature stiffness-index heuristic
  5. Population prior fusion (~118/76 resting adult)
"""
from __future__ import annotations

import json

import numpy as np
import torch

from bp_features import extract_bp_features, _peak_features, _bandpass
from bp_model import BPRegressor
from bp_wavenet import BPWaveNet, normalize_waveform, WAVE_LEN
from paths import BP_MODEL_PATH, BP_NORM_PATH, BP_WAVENET_PATH

_MODEL: dict = {
    "mlp": None, "cnn": None, "device": None,
    "ready": False, "norm": None, "error": None,
}

BASE_SBP = 118.0
BASE_DBP = 76.0
SBP_MIN, SBP_MAX = 95, 175
DBP_MIN, DBP_MAX = 58, 105


def _load_norm():
    if not BP_NORM_PATH.exists():
        return None
    with open(BP_NORM_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preload_bp_model(device=None):
    if _MODEL["ready"]:
        return True
    try:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not BP_MODEL_PATH.exists() or not BP_WAVENET_PATH.exists():
            from train_bp_model import train_and_save
            train_and_save()

        mlp = BPRegressor()
        mlp.load_state_dict(torch.load(str(BP_MODEL_PATH), map_location=device, weights_only=False))
        mlp.eval().to(device)

        cnn = BPWaveNet()
        cnn.load_state_dict(torch.load(str(BP_WAVENET_PATH), map_location=device, weights_only=False))
        cnn.eval().to(device)

        _MODEL["mlp"] = mlp
        _MODEL["cnn"] = cnn
        _MODEL["device"] = device
        _MODEL["norm"] = _load_norm()
        _MODEL["ready"] = True
        _MODEL["error"] = None
        return True
    except Exception as exc:
        _MODEL["error"] = str(exc)
        _MODEL["ready"] = False
        return False


def bp_model_available():
    return bool(_MODEL["ready"])


def bp_model_error():
    return _MODEL.get("error")


def _denorm_residual(pred, norm):
    """Model outputs normalized residuals from population baseline."""
    if norm and norm.get("residual_mode"):
        sbp = BASE_SBP + float(pred[0]) * norm["sbp_std"]
        dbp = BASE_DBP + float(pred[1]) * norm["dbp_std"]
    elif norm:
        sbp = float(pred[0]) * norm["sbp_std"] + norm["sbp_mean"]
        dbp = float(pred[1]) * norm["dbp_std"] + norm["dbp_mean"]
    else:
        sbp = BASE_SBP + float(pred[0]) * 12.0
        dbp = BASE_DBP + float(pred[1]) * 8.0
    return sbp, dbp


def _pwa_estimate(wave, fs, hr):
    """Pulse wave analysis heuristic (stiffness / reflection index)."""
    hr = float(hr) if hr is not None else 72.0
    filt = _bandpass(wave, fs)
    peak = _peak_features(filt, fs)
    if peak is None:
        return BASE_SBP + 0.22 * (hr - 70), BASE_DBP + 0.12 * (hr - 70)

    si = float(peak["stiffness_index"])
    ri = float(peak["reflection_index"])
    asp = float(peak["asp"])

    # literature-inspired mapping to mmHg
    sbp = 102.0 + 8.5 * np.tanh(si / 3.5) + 0.18 * (hr - 70) + 6.0 * ri
    dbp = 68.0 + 4.2 * np.tanh(si / 4.0) + 0.10 * (hr - 70) + 3.0 * ri
    sbp += np.clip(asp * 2.5, -4, 8)
    return float(sbp), float(dbp)


def _fuse_bp(candidates, weights):
    """Weighted robust fusion — reject floor outliers."""
    pairs = [(float(s), float(d), float(w)) for s, d, w in candidates if s and d]
    if not pairs:
        return None, None

    sbps = np.array([p[0] for p in pairs])
    dbps = np.array([p[1] for p in pairs])
    wts = np.array([p[2] for p in pairs])
    wts = wts / (wts.sum() + 1e-6)

    med_s = float(np.median(sbps))
    cluster = [i for i, s in enumerate(sbps) if abs(s - med_s) <= 14]
    if len(cluster) >= 2:
        cw = wts[cluster]
        cw = cw / cw.sum()
        sbp = float(np.dot(sbps[cluster], cw))
        dbp = float(np.dot(dbps[cluster], cw))
    else:
        sbp = float(np.dot(sbps, wts))
        dbp = float(np.dot(dbps, wts))

    return sbp, dbp


def _finalize(sbp, dbp, quality, hr):
    sbp = int(round(np.clip(sbp, SBP_MIN, SBP_MAX)))
    dbp = int(round(np.clip(dbp, DBP_MIN, DBP_MAX)))
    if sbp <= dbp + 12:
        dbp = max(DBP_MIN, sbp - 44)
    pp = sbp - dbp
    map_val = int(round(dbp + pp / 3.0))

    score = float(quality)
    if hr and 55 <= hr <= 120:
        score += 0.12
    if 100 <= sbp <= 140 and 65 <= dbp <= 90:
        score += 0.1
    conf = "medium" if score >= 0.65 else ("low" if score >= 0.45 else "very_low")

    return {
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "map": map_val,
        "pulse_pressure": pp,
        "bp_confidence": conf,
        "bp_quality": round(quality, 2),
        "bp_useful": conf in ("medium", "low"),
        "bp_disclaimer": (
            "Wellness estimate from facial pulse wave (rPPG) — not a medical cuff. "
            "For best accuracy, compare with a blood pressure monitor."
        ),
    }


def estimate_bp(
    waveform,
    fs,
    hr=None,
    breathing_rate=None,
    green_trace=None,
    age=None,
    sex=None,
):
    if waveform is None or len(waveform) < 60:
        return None

    wave = np.asarray(waveform, dtype=float).flatten()
    fs = max(float(fs), 8.0)

    features, quality = extract_bp_features(
        wave, fs, hr=hr, breathing_rate=breathing_rate,
        green_trace=green_trace, age=age, sex=sex,
    )
    if quality < 0.28:
        return None

    if not _MODEL["ready"]:
        if not preload_bp_model():
            sbp, dbp = _pwa_estimate(wave, fs, hr)
            return _finalize(sbp, dbp, quality, hr)

    device = _MODEL["device"]
    norm = _MODEL.get("norm")
    candidates = []

    sbp_pwa, dbp_pwa = _pwa_estimate(wave, fs, hr)
    candidates.append((sbp_pwa, dbp_pwa, 1.2))

    if features is not None:
        x = torch.from_numpy(features.reshape(1, -1)).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            pred_mlp = _MODEL["mlp"](x).squeeze(0).cpu().numpy()
        sbp_m, dbp_m = _denorm_residual(pred_mlp, norm)
        candidates.append((sbp_m, dbp_m, 1.5))

    w_tensor = normalize_waveform(wave, WAVE_LEN)
    if w_tensor is not None:
        wt = w_tensor.to(device)
        with torch.inference_mode():
            pred_cnn = _MODEL["cnn"](wt).squeeze(0).cpu().numpy()
        sbp_c, dbp_c = _denorm_residual(pred_cnn, norm)
        candidates.append((sbp_c, dbp_c, 2.0))

    candidates.append((BASE_SBP + 0.15 * ((hr or 72) - 72), BASE_DBP + 0.08 * ((hr or 72) - 72), 0.6))

    fused = _fuse_bp(candidates, None)
    if fused[0] is None:
        return None
    # blend toward PWA+CNN if fused suspiciously low (floor clamp bug)
    sbp, dbp = fused
    if sbp < 100 and sbp_pwa > 105:
        sbp = 0.35 * sbp + 0.65 * sbp_pwa
        dbp = 0.35 * dbp + 0.65 * dbp_pwa

    return _finalize(sbp, dbp, quality, hr)
