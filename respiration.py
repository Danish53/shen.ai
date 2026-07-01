"""
Respiratory rate estimation from 1-D motion or color signals.
Valid range: 6–30 breaths/min (0.1–0.5 Hz).
"""
import numpy as np
from scipy.signal import butter, filtfilt

RR_MIN = 6.0
RR_MAX = 30.0
FREQ_MIN = 0.1
FREQ_MAX = 0.5


def bandpass(signal, fs, low=FREQ_MIN, high=FREQ_MAX, order=2):
    """Bandpass filter signal to the respiratory frequency band."""
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


def estimate_rr_fft(signal, fs):
    """Estimate RR (breaths/min) from FFT peak in 0.1–0.5 Hz."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < fs * 8:
        return None
    filtered = bandpass(signal - np.mean(signal), fs)
    n = len(filtered)
    fft_data = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    if not np.any(mask):
        return None
    band_fft = fft_data.copy()
    band_fft[~mask] = 0
    peak_freq = freqs[np.argmax(band_fft)]
    rr = peak_freq * 60.0
    if RR_MIN <= rr <= RR_MAX:
        return float(rr)
    return None


def estimate_rr_autocorr(signal, fs):
    """Estimate RR (breaths/min) from autocorrelation peak lag."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < fs * 8:
        return None
    filtered = bandpass(signal - np.mean(signal), fs)
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
    if RR_MIN <= rr <= RR_MAX:
        return float(rr)
    return None


def fuse_rr(estimates):
    """Fuse multiple RR estimates; returns rounded breaths/min or None."""
    valid = [
        e for e in estimates
        if e is not None and RR_MIN <= float(e) <= RR_MAX
    ]
    if not valid:
        return None
    return round(float(np.median(valid)))


def estimate_rr_robust(signal, fs):
    """Robust RR estimate using FFT and autocorrelation."""
    estimates = []
    for fn in (estimate_rr_fft, estimate_rr_autocorr):
        try:
            rr = fn(signal, fs)
            if rr is not None:
                estimates.append(rr)
        except (ValueError, TypeError):
            pass
    return fuse_rr(estimates)
