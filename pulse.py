import numpy as np
from scipy.signal import butter, filtfilt
from cdf import CDF
from asf import ASF

PRE_STEP_ASF = False
PRE_STEP_CDF = False


class Pulse():
    def __init__(self, framerate, signal_size, batch_size, image_size=256):
        self.framerate = float(framerate)
        self.signal_size = signal_size
        self.batch_size = batch_size
        self.minFreq = 0.88   # ~53 BPM — resting adult floor
        self.maxFreq = 2.8    # ~168 BPM
        self.fft_spec = []

    @staticmethod
    def bandpass_hr(signal, fs, low=0.92, high=2.35, order=2):
        """Keep only pulse band before FFT — reduces breathing/motion drift."""
        y = np.asarray(signal, dtype=float).flatten()
        if len(y) < 16:
            return y - np.mean(y)
        fs = max(float(fs), 8.0)
        nyq = 0.5 * fs
        lo = max(low / nyq, 1e-5)
        hi = min(high / nyq, 0.999)
        if lo >= hi:
            return y - np.mean(y)
        b, a = butter(order, [lo, hi], btype="band")
        pad = 3 * max(len(a), len(b))
        if len(y) < pad:
            return y - np.mean(y)
        return filtfilt(b, a, y - np.mean(y))

    def get_pulse(self, mean_rgb):
        mean_rgb = np.asarray(mean_rgb)
        n = len(mean_rgb)
        if n < 8:
            return np.zeros(max(n, 1))

        seg_t = min(3.2, max(1.8, n / max(self.framerate, 8.0)))
        l = max(16, min(int(self.framerate * seg_t), n))
        H = np.zeros(n)

        B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]

        for t in range(0, max(0, n - l + 1)):
            chunk = mean_rgb[t:t + l, :]
            if chunk.shape[0] < 2:
                continue

            C = chunk.T

            if PRE_STEP_CDF and n >= 48:
                C = CDF(C, B)

            if PRE_STEP_ASF:
                C = ASF(C)

            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color) + np.eye(3) * 1e-4
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv, C)
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
            S = np.matmul(projection_matrix, Cn)
            denom = np.std(S[1, :])
            if denom == 0 or not np.isfinite(denom):
                continue
            std = np.array([1, np.std(S[0, :]) / denom])
            P = np.matmul(std, S)
            seg_len = chunk.shape[0]
            H[t:t + seg_len] = H[t:t + seg_len] + (P - np.mean(P))
        return H

    def _pick_peak_hr(self, fft_band, freq):
        """Robust peak — rejects breathing/drift below 1 Hz and FPS-scaled harmonics."""
        band = (freq >= self.minFreq) & (freq <= self.maxFreq)
        if not np.any(band):
            return None

        masked = fft_band.copy().astype(float)
        masked[~band] = 0
        if masked.max() <= 0:
            return None

        peaks = []
        work = masked.copy()
        for _ in range(6):
            peak_idx = int(np.argmax(work))
            peak_mag = float(work[peak_idx])
            if peak_mag <= 0:
                break
            peak_freq = float(freq[peak_idx])
            peaks.append((peak_freq, peak_mag))
            lo = max(0, peak_idx - 2)
            hi = min(len(work), peak_idx + 3)
            work[lo:hi] = 0

        if not peaks:
            return None

        best_freq = peaks[0][0]
        best_score = -1.0
        for peak_freq, peak_mag in peaks:
            score = peak_mag
            if 1.0 <= peak_freq <= 1.55:
                score *= 1.4
            elif peak_freq < 1.0:
                score *= 0.55
            elif peak_freq > 1.72:
                score *= 0.75
            for other_f, other_m in peaks:
                if other_f <= peak_freq:
                    continue
                ratio = other_f / max(peak_freq, 1e-6)
                if 1.45 < ratio < 1.85 and 55 <= peak_freq * 60 <= 98:
                    score *= 1.25
                if 1.55 < ratio < 2.25 and other_m > 0.5 * peak_mag:
                    score *= 0.6
            if score > best_score:
                best_score = score
                best_freq = peak_freq

        peak_freq = best_freq
        peak_idx = int(np.argmin(np.abs(freq - peak_freq)))
        peak_mag = float(masked[peak_idx])

        if peak_freq < 1.05 and peak_mag > 0:
            for mult in (2.0, 3.0):
                hf = peak_freq * mult
                if hf > self.maxFreq:
                    break
                hi = int(np.argmin(np.abs(freq - hf)))
                hm = float(masked[hi])
                if hm > 0.42 * peak_mag:
                    peak_freq = float(freq[hi])
                    peak_mag = hm
                    break

        if peak_freq > 1.85 and peak_mag > 0:
            half_freq = peak_freq / 2.0
            if half_freq >= self.minFreq:
                half_idx = int(np.argmin(np.abs(freq - half_freq)))
                half_mag = float(masked[half_idx])
                if half_mag > 0.55 * peak_mag:
                    peak_freq = float(freq[half_idx])

        hr = peak_freq * 60.0
        if not np.isfinite(hr):
            return None
        return hr

    def hr_from_autocorr(self, signal):
        """Autocorrelation HR — cross-check for FFT mistakes."""
        signal = np.asarray(signal, dtype=float).flatten()
        n = len(signal)
        if n < 30:
            return None
        y = signal - np.mean(signal)
        std = float(np.std(y))
        if std < 1e-9:
            return None
        y = y / std
        ac = np.correlate(y, y, mode='full')[n - 1:]
        if ac[0] <= 0:
            return None
        ac = ac / ac[0]

        fs = max(self.framerate, 8.0)
        min_lag = max(1, int(fs / self.maxFreq))
        max_lag = min(int(fs / self.minFreq), n - 1)
        if max_lag <= min_lag:
            return None
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag + 1]))
        hr = (60.0 * fs) / lag
        if 55 <= hr <= 175:
            return float(hr)
        return None

    def get_rfft_hr(self, signal):
        signal = np.asarray(signal, dtype=float).flatten()
        signal_size = len(signal)
        if signal_size < 20:
            return None

        window = np.hanning(signal_size)
        centered = self.bandpass_hr(signal, self.framerate)
        weighted = centered * window

        fft_data = np.abs(np.fft.rfft(weighted))
        freq = np.fft.rfftfreq(signal_size, 1.0 / self.framerate)

        hr = self._pick_peak_hr(fft_data, freq)
        if hr is not None:
            self.fft_spec.append(fft_data)
        return hr

    @staticmethod
    def hr_from_green_trace(green, fs):
        """Fallback HR from green channel with harmonic rejection."""
        g = np.asarray(green, dtype=float).flatten()
        if len(g) < 24:
            return None
        g = g - np.mean(g)
        g = g / (np.std(g) + 1e-6)
        g = Pulse.bandpass_hr(g, max(fs, 8.0))
        n = len(g)
        window = np.hanning(n)
        fft_data = np.abs(np.fft.rfft(g * window))
        freq = np.fft.rfftfreq(n, 1.0 / max(fs, 8.0))
        pulse = Pulse(max(fs, 8.0), n, 5)
        hr = pulse._pick_peak_hr(fft_data, freq)
        if hr is None or not np.isfinite(hr):
            return None
        hr = float(hr)
        if 52 <= hr <= 180:
            return hr
        return None
