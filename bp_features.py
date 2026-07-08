"""Pulse-wave morphology features for cuffless BP estimation from rPPG."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _bandpass(y, fs, low=0.7, high=4.0):
    y = np.asarray(y, dtype=float).flatten()
    if len(y) < 20:
        return y - np.mean(y)
    fs = max(float(fs), 8.0)
    nyq = 0.5 * fs
    lo, hi = max(low / nyq, 1e-5), min(high / nyq, 0.999)
    if lo >= hi:
        return y - np.mean(y)
    b, a = butter(2, [lo, hi], btype="band")
    pad = 3 * max(len(a), len(b))
    if len(y) < pad:
        return y - np.mean(y)
    return filtfilt(b, a, y - np.mean(y))


def _safe_stats(x):
    x = np.asarray(x, dtype=float).flatten()
    if len(x) < 2:
        return 0.0, 0.0, 0.0, 0.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-9:
        return m, 0.0, 0.0, 0.0
    z = (x - m) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return m, s, skew, kurt


def _peak_features(wave, fs):
    wave = np.asarray(wave, dtype=float).flatten()
    if len(wave) < 30:
        return None

    min_dist = max(3, int(fs * 0.35))
    peaks, props = find_peaks(wave, distance=min_dist, prominence=np.std(wave) * 0.15)
    if len(peaks) < 3:
        return None

    prominences = props.get("prominences", np.ones(len(peaks)))
    intervals = np.diff(peaks) / fs
    valid_iv = intervals[(intervals > 0.35) & (intervals < 1.4)]
    if len(valid_iv) < 2:
        return None

    hr_from_peaks = 60.0 / float(np.median(valid_iv))
    sdnn = float(np.std(valid_iv)) * 1000.0
    rmssd = float(np.sqrt(np.mean(np.diff(valid_iv) ** 2))) * 1000.0 if len(valid_iv) > 2 else 0.0

    seg_starts = peaks[:-1]
    seg_ends = peaks[1:]
    asp_list, dip_list, width_list = [], [], []
    for s, e in zip(seg_starts, seg_ends):
        seg = wave[s:e + 1]
        if len(seg) < 4:
            continue
        asp = float(seg.max() - seg.min())
        asp_list.append(asp)
        peak_i = int(np.argmax(seg))
        if peak_i < len(seg) - 2:
            post = seg[peak_i:]
            dip = float(seg[peak_i] - post.min()) if len(post) > 1 else 0.0
            dip_list.append(dip / (asp + 1e-6))
        half = asp * 0.5 + seg.min()
        above = seg >= half
        width_list.append(float(np.sum(above)) / fs)

    if not asp_list:
        return None

    asp = float(np.median(asp_list))
    ri = float(np.median(dip_list)) if dip_list else 0.0
    pw = float(np.median(width_list)) if width_list else 0.0
    si = asp / max(pw, 1e-3)

    return {
        "hr_peak": hr_from_peaks,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "asp": asp,
        "reflection_index": ri,
        "pulse_width": pw,
        "stiffness_index": si,
    }


def extract_bp_features(
    waveform,
    fs,
    hr=None,
    breathing_rate=None,
    green_trace=None,
    age=None,
    sex=None,
):
    """
    Build fixed-length feature vector for BP regressor.
    Returns (features[18], quality_score 0-1) or (None, 0).
    """
    wave = _bandpass(waveform, fs)
    if len(wave) < 60:
        return None, 0.0

    mean_v, std_v, skew_v, kurt_v = _safe_stats(wave)
    peak = _peak_features(wave, fs)
    if peak is None:
        return None, 0.0

    hr_val = float(hr) if hr is not None else peak["hr_peak"]
    br_val = float(breathing_rate) if breathing_rate is not None else 14.0
    age_val = float(age) if age is not None else 45.0
    sex_val = 1.0 if str(sex).upper() in ("F", "FEMALE", "1") else 0.0

    ac_dc = 0.0
    if green_trace is not None and len(green_trace) >= 30:
        g = np.asarray(green_trace, dtype=float)
        dc = float(np.mean(g))
        ac = float(np.std(g))
        ac_dc = ac / (dc + 1e-6)

    fft = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
    freqs = np.fft.rfftfreq(len(wave), 1.0 / fs)
    band = (freqs >= 0.8) & (freqs <= 3.0)
    snr = float(fft[band].max() / (np.median(fft[band]) + 1e-6)) if np.any(band) else 0.0

    # normalized feature vector
    features = np.array([
        hr_val / 100.0,
        br_val / 25.0,
        age_val / 80.0,
        sex_val,
        mean_v,
        std_v,
        skew_v,
        kurt_v,
        peak["asp"],
        peak["reflection_index"],
        peak["pulse_width"],
        peak["stiffness_index"] / 10.0,
        peak["sdnn_ms"] / 100.0,
        peak["rmssd_ms"] / 100.0,
        ac_dc * 100.0,
        snr / 10.0,
        peak["hr_peak"] / 100.0,
        len(wave) / (fs * 20.0),
    ], dtype=np.float32)

    quality = min(1.0, 0.25 + 0.15 * min(len(wave) / (fs * 12.0), 1.0) + 0.1 * min(snr / 8.0, 1.0))
    return features, float(quality)


FEATURE_DIM = 18
