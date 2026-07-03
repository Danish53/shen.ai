"""
Respiratory rate estimation from 1-D motion or color signals.
Valid range: 6–30 breaths/min (0.1–0.5 Hz).
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

RR_MIN = 6.0
RR_MAX = 30.0
FREQ_MIN = 0.1
FREQ_MAX = 0.5


def min_rr_samples(fs, seconds=4.0):
    """Minimum samples for a breathing estimate (~4 s at nominal FPS)."""
    return max(40, int(float(fs) * seconds))


def normalize_trace(signal):
    y = np.asarray(signal, dtype=float)
    if len(y) < 4:
        return y - np.mean(y)
    std = float(np.std(y))
    if std < 1e-9:
        return y - np.mean(y)
    return (y - np.mean(y)) / std


def bandpass(signal, fs, low=FREQ_MIN, high=FREQ_MAX, order=2):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 8:
        return signal - np.mean(signal)
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return signal - np.mean(signal)
    b, a = butter(order, [low_n, high_n], btype='band')
    pad = 3 * max(len(a), len(b))
    if len(signal) < pad:
        return signal - np.mean(signal)
    return filtfilt(b, a, signal)


def detrend_linear(signal):
    y = np.asarray(signal, dtype=float)
    n = len(y)
    if n < 4:
        return y - np.mean(y)
    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)


def _parabolic_peak_freq(freqs, spectrum, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(freqs[peak_idx])
    y0, y1, y2 = spectrum[peak_idx - 1], spectrum[peak_idx], spectrum[peak_idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[peak_idx])
    delta = 0.5 * (y0 - y2) / denom
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    return float(freqs[peak_idx] + delta * df)


def _is_bin_lock_artifact(rr, n, fs):
    if n >= int(fs * 8):
        return False
    return abs(rr - 20.0) < 1.2


def _rr_from_spectrum(freqs, spectrum, n_samples, fs, strict=True):
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    if not np.any(mask):
        return None

    band_fft = spectrum.copy()
    band_fft[~mask] = 0.0
    if band_fft.max() <= 0:
        return None

    peak_idx = int(np.argmax(band_fft))
    peak_val = float(band_fft[peak_idx])
    band_median = float(np.median(band_fft[mask]))

    if strict and peak_val < max(band_median * 1.35, band_fft.max() * 0.12):
        return None

    peak_freq = _parabolic_peak_freq(freqs, band_fft, peak_idx)
    rr = peak_freq * 60.0
    if not (RR_MIN <= rr <= RR_MAX):
        return None
    if _is_bin_lock_artifact(rr, n_samples, fs):
        return None
    return float(rr)


def _spectrum_rr(signal, fs, strict=True):
    fs = max(float(fs), 3.0)
    detrended = detrend_linear(normalize_trace(signal))
    try:
        filtered = bandpass(detrended, fs)
    except (ValueError, TypeError):
        filtered = detrended

    n = len(filtered)
    if n < 24:
        return None
    window = np.hanning(n)
    weighted = filtered * window
    fft_data = np.abs(np.fft.rfft(weighted))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return _rr_from_spectrum(freqs, fft_data, n, fs, strict=strict)


def estimate_rr_fft(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < min_rr_samples(fs, seconds=4.0):
        return None
    return _spectrum_rr(signal, fs, strict=True)


def estimate_rr_autocorr(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < min_rr_samples(fs, seconds=3.5):
        return None
    filtered = bandpass(detrend_linear(normalize_trace(signal)), fs)
    n = len(filtered)
    ac = np.correlate(filtered, filtered, mode='full')
    ac = ac[n - 1:]
    if ac[0] == 0:
        return None
    ac = ac / ac[0]

    min_lag = max(1, int(fs / FREQ_MAX))
    max_lag = min(int(fs / FREQ_MIN), n - 1)
    if max_lag <= min_lag:
        return None

    lag = min_lag + int(np.argmax(ac[min_lag:max_lag + 1]))
    rr = (60.0 * fs) / lag
    if RR_MIN <= rr <= RR_MAX and not _is_bin_lock_artifact(rr, n, fs):
        return float(rr)
    return None


def estimate_rr_short(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < min_rr_samples(fs, seconds=3.0):
        return None
    return _spectrum_rr(signal, fs, strict=True)


def estimate_rr_lenient(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < min_rr_samples(fs, seconds=2.0):
        return None
    return _spectrum_rr(signal, fs, strict=False)


def estimate_rr_from_envelope(signal, fs):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < min_rr_samples(fs, seconds=3.0):
        return None
    centered = detrend_linear(normalize_trace(signal))
    envelope = np.abs(hilbert(centered))
    win = max(3, int(fs * 0.4))
    kernel = np.ones(win, dtype=float) / win
    smooth = np.convolve(envelope, kernel, mode='same')
    return estimate_rr_short(smooth, fs) or estimate_rr_lenient(smooth, fs)


def fuse_rr(estimates):
    valid = [
        float(e) for e in estimates
        if e is not None and RR_MIN <= float(e) <= RR_MAX
    ]
    if not valid:
        return None
    if len(valid) == 1:
        return round(valid[0])

    med = float(np.median(valid))
    cluster = [v for v in valid if abs(v - med) <= 7.0]
    if cluster:
        return round(float(np.median(cluster)))
    return round(med)


def estimate_rr_robust(signal, fs):
    estimates = []
    for fn in (estimate_rr_fft, estimate_rr_autocorr, estimate_rr_short):
        try:
            rr = fn(signal, fs)
            if rr is not None:
                estimates.append(rr)
        except (ValueError, TypeError):
            pass
    fused = fuse_rr(estimates)
    if fused is not None:
        return fused
    try:
        rr = estimate_rr_lenient(signal, fs)
        return round(rr) if rr is not None else None
    except (ValueError, TypeError):
        return None
