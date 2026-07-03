import numpy as np
from cdf import CDF
from asf import ASF
from numpy.linalg import inv

from scipy.fftpack import rfftfreq, rfft
import cv2
from torchvision import transforms
import pdb
from PIL import Image

PRE_STEP_ASF = False  
PRE_STEP_CDF = False

class Pulse():
    def __init__(self, framerate, signal_size, batch_size, image_size=256):
        self.framerate = float(framerate)
        self.signal_size = signal_size
        self.batch_size = batch_size
        self.minFreq = 0.9 #
        self.maxFreq = 3 #
        self.fft_spec = []
        
    def get_pulse(self, mean_rgb):
        mean_rgb = np.asarray(mean_rgb)
        n = len(mean_rgb)
        if n < 8:
            return np.zeros(max(n, 1))

        # Shorter window when fewer samples (remote scan may finalize before 80 frames).
        seg_t = min(3.2, max(1.6, n / max(self.framerate, 8.0)))
        l = max(12, min(int(self.framerate * seg_t), n))
        H = np.zeros(n)

        B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]
                
        for t in range(0, max(0, n - l + 1)):
            chunk = mean_rgb[t:t + l, :]
            if chunk.shape[0] < 2:
                continue

            # pre processing steps
            C = chunk.T

            if PRE_STEP_CDF:
                C = CDF(C, B)
           
            if PRE_STEP_ASF:
                C = ASF(C)
           
            # POS
            mean_color = np.mean(C, axis=1)
            diag_mean_color = np.diag(mean_color)
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)
            Cn = np.matmul(diag_mean_color_inv,C)
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])
            S = np.matmul(projection_matrix,Cn)
            denom = np.std(S[1, :])
            if denom == 0 or not np.isfinite(denom):
                continue
            std = np.array([1, np.std(S[0, :]) / denom])
            P = np.matmul(std, S)
            seg_len = chunk.shape[0]
            H[t:t + seg_len] = H[t:t + seg_len] + (P - np.mean(P))
        return H

    def get_rfft_hr(self, signal):
        signal = np.asarray(signal, dtype=float).flatten()
        signal_size = len(signal)
        if signal_size < 16:
            return None

        window = np.hanning(signal_size)
        centered = signal - np.mean(signal)
        weighted = centered * window

        fft_data = np.abs(np.fft.rfft(weighted))
        freq = np.fft.rfftfreq(signal_size, 1.0 / self.framerate)

        band = (freq >= self.minFreq) & (freq <= self.maxFreq)
        if not np.any(band):
            return None

        fft_band = fft_data.copy()
        fft_band[~band] = 0

        peak_idx = int(np.argmax(fft_band))
        peak_freq = float(freq[peak_idx])
        peak_mag = float(fft_band[peak_idx])

        # Prefer fundamental over 2× harmonic (common rPPG error → inflated HR).
        if peak_freq > 1.2 and peak_mag > 0:
            half_freq = peak_freq / 2.0
            if half_freq >= self.minFreq:
                half_idx = int(np.argmin(np.abs(freq - half_freq)))
                half_mag = float(fft_band[half_idx])
                if half_mag > 0.5 * peak_mag:
                    peak_freq = float(freq[half_idx])

        fft_data[peak_idx] = fft_data[peak_idx] ** 2
        self.fft_spec.append(fft_data)
        return peak_freq * 60.0
