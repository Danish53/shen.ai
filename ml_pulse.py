"""
Deep-learning pulse estimation (DeepPhys) for webcam rPPG.
Uses pretrained PURE weights from ubicomplab/rPPG-Toolbox.
"""
from __future__ import annotations

import urllib.request
from collections import deque

import cv2
import numpy as np
import torch

from deepphys_model import DeepPhys
from paths import DEEPPHYS_WEIGHTS_PATH, DEEPPHYS_WEIGHTS_URL
from pulse import Pulse
from respiration import (
    estimate_rr_fft,
    estimate_rr_from_envelope,
    estimate_rr_robust,
    fuse_rr,
    min_rr_samples,
)
from utils import moving_avg

IMG_SIZE = 72
WINDOW_FRAMES = 96
MIN_RR_WAVE_SAMPLES = 40


def _resample_trace(trace, target_len):
    """Align irregular ML waveform length to video frame timeline."""
    y = np.asarray(trace, dtype=float).flatten()
    if len(y) < 8 or target_len is None or target_len < 8:
        return y
    if len(y) == target_len:
        return y
    x_old = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, int(target_len))
    return np.interp(x_new, x_old, y)


def _motion_from_crops(prev_rgb, curr_rgb):
    """Vertical centroid shift in forehead crop — breathing motion proxy."""
    prev_g = (
        0.299 * prev_rgb[:, :, 0]
        + 0.587 * prev_rgb[:, :, 1]
        + 0.114 * prev_rgb[:, :, 2]
    )
    curr_g = (
        0.299 * curr_rgb[:, :, 0]
        + 0.587 * curr_rgb[:, :, 1]
        + 0.114 * curr_rgb[:, :, 2]
    )
    diff = np.abs(curr_g - prev_g)
    row_energy = diff.sum(axis=1)
    total = float(row_energy.sum())
    if total < 1e-6:
        return None
    y = np.arange(len(row_energy), dtype=float)
    return float(np.dot(row_energy, y) / total)


MIN_WAVE_SAMPLES = 48
INFER_EVERY_N_FRAMES = 3

_MODEL_STATE: dict = {"model": None, "device": None, "ready": False, "error": None}


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _download_weights(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEEPPHYS_WEIGHTS_URL, str(dest))


def preload_deepphys(device=None):
    """Load DeepPhys once at startup. Safe to call multiple times."""
    if _MODEL_STATE["ready"]:
        return True
    if _MODEL_STATE["error"] is not None and _MODEL_STATE["model"] is None:
        return False

    try:
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not DEEPPHYS_WEIGHTS_PATH.exists():
            _download_weights(DEEPPHYS_WEIGHTS_PATH)

        model = DeepPhys(img_size=IMG_SIZE)
        state = torch.load(str(DEEPPHYS_WEIGHTS_PATH), map_location=device, weights_only=False)
        model.load_state_dict(_strip_module_prefix(state))
        model.eval()
        model.to(device)

        _MODEL_STATE["model"] = model
        _MODEL_STATE["device"] = device
        _MODEL_STATE["ready"] = True
        _MODEL_STATE["error"] = None
        return True
    except Exception as exc:
        _MODEL_STATE["error"] = str(exc)
        _MODEL_STATE["ready"] = False
        return False


def ml_pulse_available():
    return bool(_MODEL_STATE["ready"])


def ml_pulse_error():
    return _MODEL_STATE.get("error")


def extract_face_crop(bgr, scan_mask, forehead_roi=None, chest_y0=None, size=IMG_SIZE):
    """72x72 RGB crop — forehead ROI preferred, else skin-mask bbox."""
    h, w = bgr.shape[:2]
    x0 = y0 = x1 = y1 = None

    if forehead_roi and isinstance(forehead_roi, dict):
        try:
            x0 = int(float(forehead_roi["x0"]) * w)
            x1 = int(float(forehead_roi["x1"]) * w)
            y0 = int(float(forehead_roi["y0"]) * h)
            y1 = int(float(forehead_roi["y1"]) * h)
            if chest_y0 is not None:
                y1 = min(y1, int(chest_y0))
        except (KeyError, TypeError, ValueError):
            x0 = y0 = x1 = y1 = None

    if x0 is None and scan_mask is not None and int(scan_mask.sum()) >= 30:
        ys, xs = np.where(scan_mask)
        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        side = max(bx1 - bx0, by1 - by0) * 1.35
        x0 = int(cx - side / 2)
        x1 = int(cx + side / 2)
        y0 = int(cy - side / 2)
        y1 = int(cy + side / 2)

    if x0 is None:
        return None

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, max(x0 + 12, x1))
    y1 = min(h, max(y0 + 12, y1))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None

    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)


def preprocess_deepphys_window(frames):
    """
    Match rPPG-Toolbox: DiffNormalized + Standardized → 6 channels per frame.
    frames: (n, h, w, 3) float32 RGB
    returns: (n, 6, h, w) float32
    """
    n, h, w, _ = frames.shape
    diff = np.zeros((n, h, w, 3), dtype=np.float32)
    for j in range(n - 1):
        diff[j] = (frames[j + 1] - frames[j]) / (frames[j + 1] + frames[j] + 1e-7)

    dstd = float(np.std(diff[: n - 1]))
    if dstd > 1e-8:
        diff[: n - 1] /= dstd
    diff[np.isnan(diff)] = 0.0

    std_frames = frames.astype(np.float32).copy()
    std_frames -= np.mean(std_frames)
    std_frames /= float(np.std(std_frames) + 1e-8)
    std_frames[np.isnan(std_frames)] = 0.0

    data = np.concatenate([diff, std_frames], axis=-1)
    return np.transpose(data, (0, 3, 1, 2))


class DeepPulseEstimator:
    """Streaming DeepPhys inference — one estimator per scan session."""

    def __init__(self):
        self._frames = deque(maxlen=256)
        self._waveform = deque(maxlen=900)
        self._motion_trace = deque(maxlen=900)
        self._last_infer_len = 0
        self._frame_count = 0

    def reset(self):
        self._frames.clear()
        self._waveform.clear()
        self._motion_trace.clear()
        self._last_infer_len = 0
        self._frame_count = 0

    def push_frame(self, bgr, scan_mask, forehead_roi=None, chest_y0=None):
        if not _MODEL_STATE["ready"]:
            return
        crop = extract_face_crop(bgr, scan_mask, forehead_roi, chest_y0)
        if crop is None:
            return
        if self._frames:
            motion = _motion_from_crops(self._frames[-1], crop)
            if motion is not None:
                self._motion_trace.append(motion)
        self._frames.append(crop)
        self._frame_count += 1
        if len(self._frames) < 24:
            return
        if self._frame_count % INFER_EVERY_N_FRAMES != 0:
            return
        self._run_window_inference()

    def _run_window_inference(self):
        model = _MODEL_STATE["model"]
        device = _MODEL_STATE["device"]
        n = min(len(self._frames), WINDOW_FRAMES)
        window = np.stack(list(self._frames)[-n:])

        data = preprocess_deepphys_window(window)
        tensor = torch.from_numpy(data).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            pred = model(tensor).squeeze(-1).detach().cpu().numpy()

        pred = np.asarray(pred, dtype=float).flatten()
        start = max(0, self._last_infer_len - 4)
        for val in pred[start:]:
            if np.isfinite(val):
                self._waveform.append(float(val))
        self._last_infer_len = len(pred)

    def estimate_hr(self, fs):
        if len(self._waveform) < MIN_WAVE_SAMPLES:
            return None

        fs = max(float(fs), 10.0)
        take = min(len(self._waveform), int(fs * 14))
        wave = np.asarray(list(self._waveform)[-take:], dtype=float)
        wave = moving_avg(wave, 5)
        wave = wave - np.mean(wave)

        pulse = Pulse(fs, len(wave), 5)
        hr = pulse.get_rfft_hr(wave)
        if hr is None or not np.isfinite(hr):
            return None
        hr = float(hr)
        if 52 <= hr <= 180:
            return round(hr)
        return None

    def _aligned_waveform(self, video_fs, n_frames=None):
        if len(self._waveform) < MIN_RR_WAVE_SAMPLES:
            return None, None
        fs = max(float(video_fs), 3.0)
        target = n_frames if n_frames and n_frames >= MIN_RR_WAVE_SAMPLES else None
        if target is None and self._frame_count >= MIN_RR_WAVE_SAMPLES:
            target = self._frame_count
        wave = _resample_trace(list(self._waveform), target)
        if len(wave) < MIN_RR_WAVE_SAMPLES:
            return None, None
        return wave, fs

    def estimate_rr(self, video_fs, n_frames=None):
        """
        Breathing rate from DeepPhys PPG waveform (respiratory modulation).
        """
        wave, fs = self._aligned_waveform(video_fs, n_frames)
        if wave is None:
            return None
        wave = moving_avg(wave - np.mean(wave), 7)
        estimates = []
        for fn in (estimate_rr_from_envelope, estimate_rr_robust, estimate_rr_fft):
            try:
                rr = fn(wave, fs)
                if rr is not None:
                    estimates.append(float(rr))
            except (ValueError, TypeError):
                pass
        fused = fuse_rr(estimates)
        return int(fused) if fused is not None else None

    def estimate_rr_motion(self, video_fs, n_frames=None):
        """
        Breathing rate from forehead-crop vertical motion (ML pipeline crops).
        """
        if len(self._motion_trace) < min_rr_samples(video_fs, seconds=2.5):
            return None
        fs = max(float(video_fs), 3.0)
        target = n_frames if n_frames and n_frames >= MIN_RR_WAVE_SAMPLES else self._frame_count
        motion = _resample_trace(list(self._motion_trace), target)
        if len(motion) < min_rr_samples(fs, seconds=2.5):
            return None
        estimates = []
        for fn in (estimate_rr_robust, estimate_rr_fft):
            try:
                rr = fn(motion, fs)
                if rr is not None:
                    estimates.append(float(rr))
            except (ValueError, TypeError):
                pass
        fused = fuse_rr(estimates)
        return int(fused) if fused is not None else None

    @property
    def waveform_len(self):
        return len(self._waveform)

    @property
    def motion_len(self):
        return len(self._motion_trace)
