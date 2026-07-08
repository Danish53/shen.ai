"""
Unified vitals scanner for the API.
"""
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms

from models import LinkNet34
from pulse import Pulse
from ml_pulse import DeepPulseEstimator, preload_deepphys, ml_pulse_available
from bp_estimator import estimate_bp, preload_bp_model, bp_model_available
from respiration import (
    estimate_rr_robust,
    estimate_rr_short,
    estimate_rr_lenient,
    estimate_rr_from_envelope,
    fuse_rr,
    min_rr_samples,
    detrend_linear,
)
from utils import moving_avg
from paths import MODEL_PATH

torch.set_num_threads(2)

RPPG_TARGET_FPS = 25.0
RPPG_MIN_FPS = 18.0
RPPG_MAX_FPS = 30.0

_MODEL_CACHE = {"model": None, "transform": None, "device": None}


class VitalsScanner:
    def __init__(self, duration=30, source=0, on_event=None, remote_mode=False):
        self.duration = duration
        self.source = source
        self.on_event = on_event or (lambda e: None)
        self.remote_mode = remote_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 5
        self.signal_size = self._signal_capacity(duration)
        self.pulse_framerate = RPPG_TARGET_FPS
        self.stop_flag = False

        self.signal = np.zeros((self.signal_size, 3))
        self.pulse = Pulse(self.pulse_framerate, self.signal_size, self.batch_size)
        self.hrs = []
        self.brs = []
        self.chest_green = []
        self.chest_motion = []
        self.face_green = []
        self.br_green_trace = []
        self.br_motion_trace = []
        self.face_ok_frames = 0
        self.chest_active_frames = 0
        self.frame_count = 0
        self.actual_fps = RPPG_TARGET_FPS
        self.latest_br = None
        self.latest_hr = None

        self._batch_buf = []
        self._signal_filled = 0
        self.model = None
        self.preview_mode = True
        self.scan_start_time = None
        self._scan_accumulated = 0.0
        self._last_scan_tick = None
        self._overlay_tick = 0
        self._scan_frame_count = 0
        self._scan_fps_start = None
        self._prev_overlay_alpha = None
        self._img_transform = None
        self._client_ts_buf = []
        self._last_client_ts = None
        self._client_fps_ready = False
        self._last_mask = None
        self._last_pred = None
        self._frame_i = 0
        self.ml_pulse = DeepPulseEstimator()

    def _push_signal_sample(self, bgr_mean):
        """One RGB sample per frame — avoids waiting for batches of 5."""
        row = np.asarray(bgr_mean, dtype=float).reshape(1, 3)
        if self._signal_filled >= self.signal_size:
            self.signal[:-1] = self.signal[1:]
            self.signal[-1:] = row
        else:
            self.signal[self._signal_filled] = row[0]
        self._signal_filled += 1

    def _skin_sample_mask(self, h, w, chest_y0, scan_mask, forehead_roi=None):
        """Forehead ROI is best for pulse — fall back to full face mask."""
        roi = self._mask_from_forehead_roi(h, w, chest_y0, forehead_roi)
        if roi is not None and int(roi.sum()) >= 20:
            return roi
        return scan_mask

    def _ingest_face_frame(self, frame, scan_mask, chest_y0, forehead_roi=None):
        """Per-frame skin RGB for remote scans (webcam JPEG)."""
        h, w = frame.shape[:2]
        sample_m = self._skin_sample_mask(h, w, chest_y0, scan_mask, forehead_roi)
        fy = chest_y0
        face_m = sample_m[:fy, :]
        if int(face_m.sum()) < 20:
            return False
        pix = frame[:fy][face_m]
        mean_bgr = pix.mean(axis=0)
        self.face_green.append(float(mean_bgr[1]))
        self._push_signal_sample(mean_bgr)
        self.frame_count += 1
        if self._signal_filled % 5 == 0 and self._signal_filled >= 20:
            self._try_append_hr()
        if self._signal_filled % 12 == 0:
            br = self._estimate_br(live_only=True)
            if br is not None:
                self.latest_br = br
        return True

    @classmethod
    def preload_model(cls):
        """Load linknet.pth once — reused for every scan session."""
        if _MODEL_CACHE["model"] is not None:
            return
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LinkNet34(pretrained=False)
        model.load_state_dict(
            torch.load(str(MODEL_PATH), weights_only=False, map_location=device)
        )
        model.eval()
        model.to(device)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["transform"] = transform
        _MODEL_CACHE["device"] = device
        preload_deepphys(device)
        preload_bp_model(device)

    def _push_ml_frame(self, frame, scan_mask, chest_y0, forehead_roi=None):
        if not ml_pulse_available():
            return
        self.ml_pulse.push_frame(
            frame, scan_mask, forehead_roi=forehead_roi, chest_y0=chest_y0,
        )

    @staticmethod
    def _signal_capacity(duration):
        """Hold ~full scan window for stable final FFT."""
        return max(128, min(int(RPPG_TARGET_FPS * max(duration, 10)), 750))

    def _effective_fps(self):
        """Frame rate from actual samples received — critical for FFT accuracy."""
        if self._scan_accumulated > 2.0:
            if len(self.face_green) >= 20:
                measured = len(self.face_green) / self._scan_accumulated
                if 8.0 <= measured <= 32.0:
                    return float(measured)
            if self.frame_count >= 20:
                measured = self.frame_count / self._scan_accumulated
                if 8.0 <= measured <= 32.0:
                    return float(measured)

        if self.remote_mode and self._client_fps_ready:
            fps = float(self.actual_fps)
            if 8.0 <= fps <= 32.0:
                return fps

        if not self.remote_mode and self._scan_fps_start is not None:
            elapsed = time.time() - self._scan_fps_start
            if elapsed > 0.5 and self._scan_frame_count > 10:
                fps = self._scan_frame_count / elapsed
                if 8.0 <= fps <= 32.0:
                    return float(fps)

        return RPPG_TARGET_FPS

    def note_client_timestamp(self, ts_ms):
        if self._last_client_ts is not None:
            dt = (float(ts_ms) - self._last_client_ts) / 1000.0
            if 0.02 < dt < 0.25:
                self._client_ts_buf.append(dt)
                if len(self._client_ts_buf) > 60:
                    self._client_ts_buf.pop(0)
                if len(self._client_ts_buf) >= 10:
                    median_dt = float(np.median(self._client_ts_buf))
                    self.actual_fps = 1.0 / median_dt
                    self._client_fps_ready = True
        self._last_client_ts = float(ts_ms)

    @staticmethod
    def decode_frame(jpeg_b64):
        raw = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode camera frame")
        return bgr

    def _smooth_overlay_temporal(self, alpha):
        if self._prev_overlay_alpha is None or self._prev_overlay_alpha.shape != alpha.shape:
            self._prev_overlay_alpha = alpha.copy()
            return alpha
        blended = 0.62 * self._prev_overlay_alpha + 0.38 * alpha
        self._prev_overlay_alpha = blended.copy()
        return blended

    @staticmethod
    def _build_overlay_alpha(upper_prob):
        """Soft skin-tight mask from model — feathered edges like paint on face."""
        prob = upper_prob.astype(np.float32)
        if prob.max() < 0.05:
            return prob
        soft = cv2.GaussianBlur(prob, (0, 0), 7)
        soft = np.clip((soft - 0.18) / 0.62, 0.0, 1.0)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        soft_u8 = (soft * 255).astype(np.uint8)
        soft_u8 = cv2.morphologyEx(soft_u8, cv2.MORPH_CLOSE, k7)
        return cv2.GaussianBlur(soft_u8.astype(np.float32) / 255.0, (0, 0), 4)

    def _draw_skin_scan_overlay(self, display, alpha_mask, tick):
        """Light on-skin scan — auto follows detected face, face stays clear."""
        if alpha_mask is None or float(alpha_mask.max()) < 0.08:
            return

        h, w = display.shape[:2]
        result = display.astype(np.float32)
        a3 = alpha_mask[:, :, np.newaxis]
        skin = alpha_mask > 0.22

        ys, xs = np.where(skin)
        if len(xs) < 20:
            return
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        pulse = 0.5 + 0.5 * np.sin(tick * 0.07)
        tint = result.copy()
        tint[:, :, 0] = np.minimum(result[:, :, 0] + 22 * pulse, 255)
        tint[:, :, 1] = np.minimum(result[:, :, 1] + 38 * pulse, 255)
        tint[:, :, 2] = np.minimum(result[:, :, 2] + 30 * pulse, 255)
        result = result * (1.0 - a3 * 0.09) + tint * (a3 * 0.09)

        step = max(11, (x2 - x1) // 17)
        for gy in range(y1, y2, step):
            for gx in range(x1, x2, step):
                py, px = min(gy, y2 - 1), min(gx, x2 - 1)
                if alpha_mask[py, px] < 0.28:
                    continue
                phase = tick * 0.12 + gx * 0.045 + gy * 0.035
                if np.sin(phase) > 0.5:
                    r = 1 + int(alpha_mask[py, px] > 0.55)
                    cv2.circle(result, (px, py), r, (240, 255, 250), -1, cv2.LINE_AA)

        mesh = result.copy()
        step_x = max(8, (x2 - x1) // 19)
        step_y = max(8, (y2 - y1) // 21)
        for gy in range(y1, y2, step_y):
            pts = [
                (gx, gy) for gx in range(x1, x2, step_x)
                if alpha_mask[min(gy, y2 - 1), min(gx, x2 - 1)] > 0.26
            ]
            for i in range(len(pts) - 1):
                cv2.line(mesh, pts[i], pts[i + 1], (235, 252, 255), 1, cv2.LINE_AA)
        for gx in range(x1, x2, step_x):
            pts = [
                (gx, gy) for gy in range(y1, y2, step_y)
                if alpha_mask[min(gy, y2 - 1), min(gx, x2 - 1)] > 0.26
            ]
            for i in range(len(pts) - 1):
                cv2.line(mesh, pts[i], pts[i + 1], (225, 248, 255), 1, cv2.LINE_AA)
        result = result * (1.0 - a3 * 0.20) + mesh * (a3 * 0.20)

        fh = y2 - y1
        band_y = y1 + int(((tick % 70) / 70.0) * fh)
        sweep = result.copy()
        cv2.line(sweep, (x1, band_y), (x2, band_y), (195, 242, 232), 1, cv2.LINE_AA)
        band = np.zeros((h, w), dtype=np.float32)
        cv2.line(band, (x1, band_y), (x2, band_y), 1.0, 3, cv2.LINE_AA)
        band = cv2.GaussianBlur(band, (0, 0), 5) * alpha_mask
        b3 = band[:, :, np.newaxis]
        result = result * (1.0 - b3 * 0.16) + sweep * (b3 * 0.16)

        mask_u8 = (np.clip(alpha_mask, 0, 1) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            edge = result.copy()
            cv2.drawContours(edge, contours, -1, (205, 252, 248), 1, cv2.LINE_AA)
            edge_a = (alpha_mask > 0.4).astype(np.float32)[:, :, np.newaxis] * 0.12
            result = result * (1.0 - edge_a) + edge * edge_a

        display[:] = np.clip(result, 0, 255).astype(np.uint8)

    def _update_scan_fps(self):
        if self.remote_mode:
            return
        if self.scan_start_time is None:
            return
        if self._scan_fps_start is None:
            self._scan_fps_start = time.time()
            self._scan_frame_count = 0
        self._scan_frame_count += 1
        elapsed = time.time() - self._scan_fps_start
        if elapsed > 0.5:
            self.actual_fps = self._scan_frame_count / elapsed

    def finish_scan(self):
        """Return to preview after a completed scan (keep WebSocket open)."""
        self.preview_mode = True
        self.scan_start_time = None
        self._batch_buf = []
        self._overlay_tick = 0
        self._prev_overlay_alpha = None

    def finalize_scan(self):
        """Force scan completion — compute vitals from frames collected so far."""
        if self.scan_start_time is not None and self._batch_buf and not self.remote_mode:
            pad = self.batch_size - len(self._batch_buf)
            batch = np.stack(self._batch_buf)
            if pad > 0:
                batch = np.concatenate(
                    [batch, np.zeros((pad, *batch.shape[1:]), dtype=batch.dtype)]
                )
            self.process_batch(batch)
            self._batch_buf = []
        self._try_append_hr()
        br = self._estimate_br(live_only=False)
        if br is not None:
            self.latest_br = br
        self._scan_accumulated = float(self.duration)
        return {"type": "complete", **self._final_results()}

    def abort_scan(self):
        """User cancelled mid-scan — reset without closing connection."""
        self.finish_scan()
        self._reset_vitals()

    def start_scan(self, duration=None):
        if duration:
            self.duration = duration
            self.signal_size = self._signal_capacity(duration)
        self._reset_vitals()
        self._scan_accumulated = 0.0
        self._last_scan_tick = None
        self._overlay_tick = 0
        self.preview_mode = False
        self.scan_start_time = time.time()
        self.emit("status", message="Scanning…", phase="scanning")

    def _reset_vitals(self):
        self.signal = np.zeros((self.signal_size, 3))
        self.pulse = Pulse(self.pulse_framerate, self.signal_size, self.batch_size)
        self.hrs = []
        self.brs = []
        self.chest_green = []
        self.chest_motion = []
        self.face_green = []
        self.br_green_trace = []
        self.br_motion_trace = []
        self.face_ok_frames = 0
        self.chest_active_frames = 0
        self.frame_count = 0
        self.latest_br = None
        self.latest_hr = None
        self._batch_buf = []
        self._signal_filled = 0
        self._scan_frame_count = 0
        self._scan_fps_start = None
        self._prev_overlay_alpha = None
        self._client_ts_buf = []
        self._last_client_ts = None
        self._client_fps_ready = False
        self._scan_accumulated = 0.0
        self._last_scan_tick = None
        self._overlay_tick = 0
        self._last_mask = None
        self._last_pred = None
        if hasattr(self, "ml_pulse") and self.ml_pulse is not None:
            self.ml_pulse.reset()

    @staticmethod
    def _assess_lighting(bgr, skin_mask):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if skin_mask is not None and skin_mask.any():
            pixels = gray[skin_mask.astype(bool)]
            mean_l = float(pixels.mean()) if len(pixels) > 20 else float(gray.mean())
            std_l = float(pixels.std()) if len(pixels) > 20 else float(gray.std())
        else:
            mean_l = float(gray.mean())
            std_l = float(gray.std())

        if mean_l < 45:
            return 0.2, "Too dark — turn on more light"
        if mean_l < 65:
            return 0.5, "Low light — move to brighter area"
        if mean_l > 215:
            return 0.35, "Too bright — avoid direct light on face"
        if mean_l > 185:
            return 0.65, "Slightly bright"
        if std_l < 12:
            return 0.55, "Flat lighting — add side light"
        return 1.0, "Good lighting"

    @staticmethod
    def _condition_score(face_ok, face_ratio, lighting_score):
        score = 1
        if face_ok:
            score += 2
            score += min(1, int(face_ratio * 25))
        else:
            score += max(0, int(face_ratio * 15))
        score += int(lighting_score * 2)
        return min(5, max(1, score))

    @staticmethod
    def _isolate_face_mask(upper_mask):
        contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(upper_mask)
        if not contours:
            return clean
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 80:
            cv2.drawContours(clean, [largest], -1, 1, -1)
        return clean

    def _overlay_meta(self, overlay_alpha, h, w, scanning):
        """Face ROI + skin mask for client overlay — tracks face movement."""
        if overlay_alpha is None or float(overlay_alpha.max()) < 0.08:
            return None
        skin = overlay_alpha > 0.18
        ys, xs = np.where(skin)
        if len(xs) < 20:
            return None
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad_x = max(10, int((x2 - x1) * 0.12))
        pad_y = max(12, int((y2 - y1) * 0.14))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)

        meta = {
            "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            "strength": round(float(overlay_alpha.max()), 3),
            "scanning": scanning,
            "tick": self._overlay_tick,
        }

        send_mask = scanning and self._overlay_tick % 2 == 0
        if send_mask:
            try:
                crop = overlay_alpha[y1:y2 + 1, x1:x2 + 1]
                if crop.size > 0:
                    small = cv2.resize(crop, (96, 116), interpolation=cv2.INTER_LINEAR)
                    alpha = (np.clip(small, 0, 1) * 255).astype(np.uint8)
                    bgra = np.zeros((small.shape[0], small.shape[1], 4), dtype=np.uint8)
                    bgra[:, :, 0] = (small * 60).astype(np.uint8)
                    bgra[:, :, 1] = (small * 220).astype(np.uint8)
                    bgra[:, :, 2] = (small * 200).astype(np.uint8)
                    bgra[:, :, 3] = alpha
                    ok, buf = cv2.imencode(".png", bgra)
                    if ok:
                        meta["mask_fmt"] = "png"
                        meta["mask"] = base64.b64encode(buf).decode("ascii")
            except Exception:
                pass

        return meta

    @staticmethod
    def _draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=3, arm=32):
        corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
        for px, py, dx, dy in corners:
            cv2.line(img, (px, py), (px + dx * arm, py), color, thickness, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * arm), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _smooth_face_mask(face_skin):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoothed = cv2.morphologyEx(face_skin.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        return smoothed

    def draw_overlay(self, display, chest_y0, face_skin_full, overlay_alpha, face_ok, scanning):
        h, w = display.shape[:2]
        ox1, oy1 = int(w * 0.07), int(h * 0.14)
        ox2, oy2 = int(w * 0.93), int(h * 0.80)
        has_face = overlay_alpha is not None and float(overlay_alpha.max()) > 0.08

        if scanning and has_face:
            self._draw_skin_scan_overlay(display, overlay_alpha, self._overlay_tick)
            ys, xs = np.where(overlay_alpha > 0.22)
            ix1 = max(ox1, int(xs.min()) - 14)
            iy1 = max(oy1, int(ys.min()) - 14)
            ix2 = min(ox2, int(xs.max()) + 14)
            iy2 = min(oy2, int(ys.max()) + 14)
            self._draw_corner_brackets(display, ox1, oy1, ox2, oy2, (255, 255, 255), 2, 36)
            self._draw_corner_brackets(display, ix1, iy1, ix2, iy2, (0, 0, 255), 2, 22)
        else:
            self._draw_corner_brackets(display, ox1, oy1, ox2, oy2, (0, 0, 230), 3, 36)
            if has_face and face_ok:
                ys, xs = np.where(face_skin_full)
                ix1 = max(ox1, int(xs.min()) - 14)
                iy1 = max(oy1, int(ys.min()) - 14)
                ix2 = min(ox2, int(xs.max()) + 14)
                iy2 = min(oy2, int(ys.max()) + 14)
                self._draw_corner_brackets(display, ix1, iy1, ix2, iy2, (0, 0, 255), 2, 20)

    def emit(self, event_type, **data):
        self.on_event({"type": event_type, **data})

    def load_model(self):
        self.emit("status", message="Loading AI model...", phase="loading")
        VitalsScanner.preload_model()
        self.model = _MODEL_CACHE["model"]
        self._img_transform = _MODEL_CACHE["transform"]
        self.device = _MODEL_CACHE["device"]
        self.emit("status", message="Model ready", phase="ready")

    def extract_regions(self, orig, mask_bool, mask_prob=None):
        h, w = orig.shape[:2]
        chest_y0 = int(h * 0.45)
        face_mask = mask_bool[:chest_y0, :]
        chest_mask = mask_bool[chest_y0:, :]
        if mask_prob is not None:
            chest_soft = mask_prob[chest_y0:, :] > 0.45
            if int(chest_soft.sum()) > int(chest_mask.sum()):
                chest_mask = chest_soft
        face_pixels = int(face_mask.sum())
        chest_pixels = int(chest_mask.sum())
        face_ratio = face_pixels / (face_mask.size + 1e-6)
        chest_ratio = chest_pixels / (chest_mask.size + 1e-6)
        face_ok = face_pixels > 25 and face_ratio > 0.01

        face_skin_upper = self._isolate_face_mask(face_mask.astype(np.uint8))
        face_skin_upper = self._smooth_face_mask(face_skin_upper)
        face_skin_full = np.zeros((h, mask_bool.shape[1]), dtype=np.uint8)
        face_skin_full[:chest_y0, :] = face_skin_upper

        upper_prob = mask_prob[:chest_y0] if mask_prob is not None else face_mask.astype(np.float32)
        overlay_alpha_upper = self._smooth_overlay_temporal(self._build_overlay_alpha(upper_prob))
        overlay_alpha = np.zeros((h, mask_bool.shape[1]), dtype=np.float32)
        overlay_alpha[:chest_y0, :] = overlay_alpha_upper

        info = {
            "active": False, "green": 0.0, "motion": 0.0,
            "face_ok": face_ok, "face_ratio": float(face_ratio),
            "face_skin": face_skin_full,
            "overlay_alpha": overlay_alpha,
        }
        if chest_pixels > 40 and chest_ratio > 0.015:
            chest_bgr = orig[chest_y0:][chest_mask]
            ys, _ = np.where(chest_mask)
            info["active"] = True
            info["green"] = float(chest_bgr[:, 1].mean())
            info["motion"] = float(ys.mean() + chest_y0)
        return info, chest_y0

    def _sample_breathing_frame(self, orig, mask_bool, chest_y0, overlay_alpha, forehead_roi=None):
        """Face-only breathing traces — forehead ROI preferred."""
        h, w = orig.shape[:2]
        sample_m = self._skin_sample_mask(h, w, chest_y0, mask_bool, forehead_roi)
        fy = chest_y0
        face_m = sample_m[:fy, :]
        if int(face_m.sum()) > 20:
            pix = orig[:fy][face_m]
            self.br_green_trace.append(float(pix[:, 1].mean()))
        if overlay_alpha is not None:
            upper = overlay_alpha[:fy, :]
            skin = upper > 0.16
            if skin.any():
                ys, xs = np.where(skin)
                self.br_motion_trace.append(float(ys.mean()))

    @staticmethod
    def _center_face_mask(h, w, chest_y0):
        """Fallback skin region when LinkNet mask is weak (webcam / JPEG)."""
        mask = np.zeros((h, w), dtype=bool)
        cx = w // 2
        cy = int(chest_y0 * 0.42)
        rx = max(24, int(w * 0.24))
        ry = max(24, int(chest_y0 * 0.30))
        y0, y1 = max(0, cy - ry), min(chest_y0, cy + ry)
        x0, x1 = max(0, cx - rx), min(w, cx + rx)
        ys, xs = np.ogrid[y0:y1, x0:x1]
        ellipse = ((xs - cx) ** 2) / (rx ** 2 + 1e-6) + ((ys - cy) ** 2) / (ry ** 2 + 1e-6) <= 1.0
        mask[y0:y1, x0:x1] = ellipse
        return mask

    @staticmethod
    def _mask_from_forehead_roi(h, w, chest_y0, roi):
        if not roi or not isinstance(roi, dict):
            return None
        try:
            x0 = int(float(roi.get("x0", 0)) * w)
            x1 = int(float(roi.get("x1", 1)) * w)
            y0 = int(float(roi.get("y0", 0)) * h)
            y1 = int(float(roi.get("y1", 1)) * h)
        except (TypeError, ValueError):
            return None
        y1 = min(y1, chest_y0)
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0, y0):y1, max(0, x0):x1] = True
        return mask

    def _scan_mask_for_frame(self, pred_np, chest_y0, h, w, client_face_ok, forehead_roi=None):
        roi_mask = self._mask_from_forehead_roi(h, w, chest_y0, forehead_roi)
        if roi_mask is not None and int(roi_mask.sum()) >= 40:
            return roi_mask
        ai_mask = pred_np > 0.45
        face_count = int(ai_mask[:chest_y0, :].sum())
        if face_count >= 80:
            return ai_mask
        if client_face_ok is not False:
            return self._center_face_mask(h, w, chest_y0)
        return ai_mask

    def frame_to_b64(self, bgr_frame, quality=78):
        small = cv2.resize(bgr_frame, (640, int(bgr_frame.shape[0] * 640 / bgr_frame.shape[1])))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("ascii")

    def _prepare_frame(self, orig):
        if not self.remote_mode:
            orig = cv2.flip(orig, 1)
        h0, w0 = orig.shape[:2]
        if w0 > 640:
            scale = 640 / w0
            orig = cv2.resize(orig, (640, int(h0 * scale)), interpolation=cv2.INTER_AREA)
        return orig

    def _process_remote_scan_frame(self, frame_full, h, w, chest_y0, scan_mask, client_face_ok, forehead_roi):
        """Fast browser scan path — skip LinkNet when MediaPipe forehead ROI is available."""
        self._update_scan_fps()
        now = time.time()
        tracking_ok = client_face_ok is not False

        if self._last_scan_tick is not None and tracking_ok:
            self._scan_accumulated += min(now - self._last_scan_tick, 0.4)
        self._last_scan_tick = now

        if tracking_ok:
            self.face_ok_frames += 1
            self._sample_breathing_frame(
                frame_full, scan_mask, chest_y0, None, forehead_roi,
            )
            self._push_ml_frame(frame_full, scan_mask, chest_y0, forehead_roi)
            self._ingest_face_frame(frame_full, scan_mask, chest_y0, forehead_roi)

        if self._scan_accumulated >= self.duration:
            self._try_append_hr()
            return {"type": "complete", **self._final_results()}

        remaining = max(0, int(self.duration - self._scan_accumulated))
        progress = min(100, int(((self.duration - remaining) / self.duration) * 100))
        face_ok = tracking_ok and int(scan_mask[:chest_y0, :].sum()) > 25

        return {
            "type": "vitals",
            "remaining": remaining,
            "paused": not tracking_ok,
            "face_ok": face_ok,
            "condition_score": 85 if face_ok else 40,
            "lighting_message": "",
            "lighting_ok": True,
            "progress": progress,
            "overlay": None,
            "scan_fps": round(self._effective_fps(), 1),
            "live_hr": self.latest_hr,
            "live_br": self.latest_br,
            "frame": None,
        }

    def process_frame(self, orig, client_face_ok=True, forehead_roi=None):
        """Process one camera frame — used for VPS (browser camera) and local webcam."""
        if self.model is None or self._img_transform is None:
            raise RuntimeError("Model not loaded")

        orig = self._prepare_frame(orig)
        shape = orig.shape[:2]
        scanning = not self.preview_mode and self.scan_start_time is not None
        h, w = shape[0], shape[1]
        chest_y0 = int(h * 0.45)

        if self.remote_mode and scanning:
            roi_mask = self._mask_from_forehead_roi(h, w, chest_y0, forehead_roi)
            if roi_mask is not None and int(roi_mask.sum()) >= 40:
                return self._process_remote_scan_frame(
                    orig, h, w, chest_y0, roi_mask, client_face_ok, forehead_roi,
                )
            if client_face_ok is not False:
                center_mask = self._center_face_mask(h, w, chest_y0)
                return self._process_remote_scan_frame(
                    orig, h, w, chest_y0, center_mask, client_face_ok, forehead_roi,
                )

        self._frame_i += 1
        run_model = (
            self._last_pred is None
            or not scanning
            or self._frame_i % 4 == 0
        )

        if run_model:
            rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (256, 256), cv2.INTER_LINEAR)
            tensor = self._img_transform(Image.fromarray(resized)).unsqueeze(0)
            imgs = tensor.to(dtype=torch.float, device=self.device)

            with torch.inference_mode():
                pred = self.model(imgs)
            pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            pred_np = pred.squeeze().cpu().numpy()
            self._last_pred = pred_np
        else:
            pred_np = self._last_pred

        mask_bool = pred_np > 0.45

        frame_full = orig.copy()
        h, w = shape[0], shape[1]
        region_info, chest_y0 = self.extract_regions(frame_full, mask_bool, pred_np)
        scan_mask = self._scan_mask_for_frame(
            pred_np, chest_y0, h, w, client_face_ok, forehead_roi,
        )
        face_skin = region_info["face_skin"]
        overlay_alpha = region_info["overlay_alpha"]
        lighting_score, lighting_msg = self._assess_lighting(frame_full, face_skin)

        if scanning:
            self._update_scan_fps()
            now = time.time()
            tracking_ok = client_face_ok is not False

            if self._last_scan_tick is not None and tracking_ok:
                self._scan_accumulated += min(now - self._last_scan_tick, 0.4)
            self._last_scan_tick = now

            if tracking_ok:
                self.face_ok_frames += 1
                self._sample_breathing_frame(
                    frame_full, scan_mask, chest_y0, overlay_alpha, forehead_roi,
                )
                self._push_ml_frame(frame_full, scan_mask, chest_y0, forehead_roi)
                if self.remote_mode:
                    self._ingest_face_frame(frame_full, scan_mask, chest_y0, forehead_roi)
                else:
                    masked = frame_full.copy()
                    masked[~scan_mask] = 0
                    self._batch_buf.append(masked)
                    self.frame_count += 1

                    if len(self._batch_buf) >= self.batch_size:
                        batch = np.stack(self._batch_buf[: self.batch_size])
                        self._batch_buf = self._batch_buf[self.batch_size :]
                        self.process_batch(batch)

            if self._scan_accumulated >= self.duration:
                if self._batch_buf and not self.remote_mode:
                    pad = self.batch_size - len(self._batch_buf)
                    batch = np.stack(self._batch_buf)
                    if pad > 0:
                        batch = np.concatenate(
                            [batch, np.zeros((pad, *batch.shape[1:]), dtype=batch.dtype)]
                        )
                    self.process_batch(batch)
                    self._batch_buf = []
                self._try_append_hr()
                return {"type": "complete", **self._final_results()}
            remaining = max(0, int(self.duration - self._scan_accumulated))
            paused = not tracking_ok
        else:
            remaining = self.duration
            paused = False

        self._overlay_tick += 1

        condition = self._condition_score(
            region_info["face_ok"], region_info["face_ratio"], lighting_score,
        )
        progress = 0
        if scanning:
            progress = min(100, int(((self.duration - remaining) / self.duration) * 100))

        overlay = self._overlay_meta(overlay_alpha, shape[0], shape[1], scanning)

        payload = {
            "remaining": remaining,
            "paused": paused if scanning else False,
            "face_ok": region_info["face_ok"] and client_face_ok is not False,
            "condition_score": condition,
            "lighting_message": lighting_msg,
            "lighting_ok": lighting_score >= 0.65,
            "progress": progress,
            "overlay": overlay,
            "scan_fps": round(self._effective_fps(), 1) if scanning else None,
            "live_hr": self.latest_hr,
            "live_br": self.latest_br,
            "frame": None,
        }

        event_type = "vitals" if scanning else "preview"
        payload["type"] = event_type
        if not self.remote_mode:
            self.emit(event_type, **{k: v for k, v in payload.items() if k != "type"})
        return payload

    def process_batch(self, batch):
        """Same logic as terminal process_mask.process_signal + compute_mean."""
        h, w = batch.shape[1], batch.shape[2]
        chest_y0 = int(h * 0.45)
        face_batch = batch[:, :chest_y0, :, :]
        face_area = chest_y0 * w
        face_skin = (face_batch != 0).any(axis=3)
        face_pixels = face_skin.sum(axis=(1, 2))
        avg_face = face_pixels.mean()
        face_ratio = avg_face / (face_area + 1e-6)

        if avg_face <= 15 or face_ratio <= 0.006:
            return

        means = np.zeros((self.batch_size, 3))
        valid = 0
        for fi in range(self.batch_size):
            if face_pixels[fi] > 0:
                means[fi] = face_batch[fi][face_skin[fi]].mean(axis=0)
                self.face_green.append(float(means[fi, 1]))
                valid += 1
        if valid == 0:
            return

        b = self.batch_size
        if self._signal_filled >= self.signal_size:
            self.signal[:-b] = self.signal[b:]
            self.signal[-b:] = means
        else:
            end = min(self._signal_filled + b, self.signal_size)
            self.signal[self._signal_filled:end] = means[: end - self._signal_filled]
        self._signal_filled += b

        self._try_append_hr()

        br = self._estimate_br(live_only=True)
        if br is not None:
            self.latest_br = br

    def _valid_signal_rows(self, filled=None):
        cap = min(self._signal_filled, self.signal_size)
        n = cap if filled is None else min(int(filled), cap)
        if n <= 0:
            return None
        rows = self.signal[:n]
        mask = np.any(rows > 0.5, axis=1)
        if mask.sum() < 8:
            return None
        return rows[mask]

    def _fuse_hr_estimates(self, estimates, hr_ml=None):
        """Reject outliers and resolve 2× harmonic disagreements (e.g. 118 vs 72)."""
        vals = sorted({round(float(e)) for e in estimates if e is not None and np.isfinite(e)})
        if not vals:
            return None
        if len(vals) == 1:
            v = vals[0]
            return v if 52 <= v <= 180 else None

        resolved = []
        used = set()
        for i, a in enumerate(vals):
            if i in used:
                continue
            harmonic_partner = None
            for j, b in enumerate(vals):
                if j <= i or j in used:
                    continue
                ratio = b / max(a, 1e-6)
                if 1.55 < ratio < 2.2 or 0.45 < ratio < 0.65:
                    harmonic_partner = b
                    used.add(j)
                    break
                if 1.48 < ratio < 1.82:
                    harmonic_partner = b
                    used.add(j)
                    break
            if harmonic_partner is not None:
                pair = sorted([a, harmonic_partner])
                low, high = pair[0], pair[1]
                ratio = high / max(low, 1e-6)
                if 1.48 < ratio < 1.82 and 55 <= low <= 100:
                    resolved.append(round((low * high) ** 0.5))
                elif 52 <= low <= 108 and high > low * 1.45:
                    resolved.append(low)
                elif 52 <= high <= 140:
                    resolved.append(high)
                else:
                    resolved.append(round(float(np.median(pair))))
                used.add(i)
            else:
                resolved.append(a)
                used.add(i)

        if not resolved:
            resolved = vals

        best_cluster = [resolved[0]]
        for anchor in resolved:
            cluster = [v for v in resolved if abs(v - anchor) <= 12]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if hr_ml is not None and 52 <= hr_ml <= 140:
            ml_cluster = [v for v in best_cluster if abs(v - hr_ml) <= 12]
            if ml_cluster:
                return round(float(np.median(ml_cluster)))
            if len(best_cluster) == 1:
                return round(hr_ml)

        if len(best_cluster) < max(1, len(resolved) // 2) and hr_ml is not None:
            if 52 <= hr_ml <= 140:
                return round(hr_ml)

        hr = round(float(np.median(best_cluster)))
        return hr if 52 <= hr <= 180 else None

    def _estimate_hr_from_rgb(self, rgb_means, fs_override=None):
        if rgb_means is None or len(rgb_means) < 20:
            return None
        fs = max(float(fs_override) if fs_override else self._effective_fps(), 10.0)
        self.pulse.framerate = fs
        old_sz = self.pulse.signal_size
        estimates = []
        hr_ml = None
        try:
            self.pulse.signal_size = len(rgb_means)
            wave = moving_avg(self.pulse.get_pulse(rgb_means), 6)
            hr_pos = self.pulse.get_rfft_hr(wave)
            if hr_pos is not None and np.isfinite(hr_pos):
                estimates.append(float(hr_pos))

            hr_ac = self.pulse.hr_from_autocorr(wave)
            if hr_ac is not None:
                estimates.append(float(hr_ac))

            green = rgb_means[:, 1].astype(float)
            hr_green = Pulse.hr_from_green_trace(green, fs)
            if hr_green is not None:
                estimates.append(float(hr_green))

            if ml_pulse_available() and self.ml_pulse.waveform_len >= 60:
                hr_ml = self.ml_pulse.estimate_hr(fs)
                if hr_ml is not None:
                    estimates.append(float(hr_ml))
                    estimates.append(float(hr_ml))
        finally:
            self.pulse.signal_size = old_sz

        if not estimates:
            return None

        if hr_ml is not None and self.ml_pulse.waveform_len >= 100:
            agree = [e for e in estimates if abs(e - hr_ml) <= 8]
            if len(agree) >= 2:
                return round(float(np.median(agree)))

        return self._fuse_hr_estimates(estimates, hr_ml=hr_ml)

    def _estimate_hr_sliding(self, rgb, fs):
        """Multiple 12 s windows — median of agreeing segments reduces 78/91 jumps."""
        fs = max(float(fs), 10.0)
        skip = min(len(rgb) - 50, max(0, int(fs * 5.0)))
        trimmed = rgb[skip:] if skip > 0 else rgb
        win = max(50, int(fs * 12))
        step = max(20, int(fs * 4))
        if len(trimmed) < win:
            return self._estimate_hr_from_rgb(trimmed, fs_override=fs), []

        window_hrs = []
        for start in range(0, len(trimmed) - win + 1, step):
            chunk = trimmed[start : start + win]
            hr = self._estimate_hr_from_rgb(chunk, fs_override=fs)
            if hr is not None:
                window_hrs.append(int(hr))

        if not window_hrs:
            return None, []

        med = float(np.median(window_hrs))
        cluster = [h for h in window_hrs if abs(h - med) <= 7]
        if len(cluster) >= 2:
            return round(float(np.median(cluster))), window_hrs
        return round(med), window_hrs

    def _estimate_hr_full_window(self):
        """Final HR — sliding windows + full trace + ML consensus."""
        rgb = self._valid_signal_rows()
        if rgb is None or len(rgb) < 40:
            if len(self.face_green) >= 40:
                g = np.asarray(self.face_green, dtype=float)
                rgb = np.column_stack([g, g, g])
            else:
                return None
        fs = self._effective_fps()
        if self._scan_accumulated > 2.0 and len(self.face_green) >= 40:
            fs = len(self.face_green) / self._scan_accumulated

        sliding_hr, window_hrs = self._estimate_hr_sliding(rgb, fs)
        full_hr = self._estimate_hr_from_rgb(
            rgb[min(len(rgb) - 50, max(0, int(fs * 5.0))):],
            fs_override=fs,
        )
        hr_ml = None
        if ml_pulse_available() and self.ml_pulse.waveform_len >= 40:
            hr_ml = self.ml_pulse.estimate_hr(fs)

        candidates = [h for h in (sliding_hr, full_hr, hr_ml) if h is not None]
        if not candidates:
            return None

        if hr_ml is not None and self.ml_pulse.waveform_len >= 100:
            agree = [h for h in candidates if abs(h - hr_ml) <= 7]
            if len(agree) >= 2:
                return round(float(np.median(agree)))

        if sliding_hr is not None and full_hr is not None and abs(sliding_hr - full_hr) <= 8:
            return round(float(np.median([sliding_hr, full_hr])))

        if sliding_hr is not None and window_hrs:
            return sliding_hr
        return round(float(np.median(candidates)))

    def _try_append_hr(self):
        if self.frame_count < 25 and len(self.face_green) < 25:
            return
        rgb = self._valid_signal_rows()
        if rgb is None:
            return
        hr = self._estimate_hr_from_rgb(rgb)
        if hr is None:
            return
        keep = True
        if len(self.hrs) >= 3:
            med = float(np.median(self.hrs[-12:]))
            if abs(hr - med) > 12:
                keep = False
            if hr < med - 18 and hr < 62:
                keep = False
            ratio = hr / max(med, 1e-6)
            if ratio > 1.55 or ratio < 0.62:
                keep = False
        if keep:
            if len(self.hrs) > 300:
                self.hrs.pop(0)
            self.hrs.append(hr)
            recent = [h for h in self.hrs[-8:] if 52 <= h <= 140]
            if recent:
                self.latest_hr = round(float(np.median(recent)))
            elif len(self.hrs) > 2:
                self.latest_hr = round(float(np.median(self.hrs[-5:])))
            else:
                self.latest_hr = hr

    def _trim_trace(self, trace, fs, skip_seconds=1.5):
        arr = np.asarray(trace, dtype=float)
        if len(arr) < 30:
            return arr
        skip = max(0, int(fs * skip_seconds))
        if len(arr) <= skip + 40:
            return arr
        return arr[skip:]

    def _valid_green_trace(self):
        """Green channel samples from skin-masked frames only."""
        rgb = self._valid_signal_rows()
        if rgb is not None and len(rgb) >= 20:
            return rgb[:, 1].astype(float)
        if len(self.face_green) >= 20:
            return np.asarray(self.face_green, dtype=float)
        return None

    def _br_from_signal_green(self, fs):
        green = self._valid_green_trace()
        if green is None or len(green) < min_rr_samples(fs, seconds=3.0):
            if len(self.face_green) >= min_rr_samples(fs, seconds=3.0):
                green = np.asarray(self.face_green, dtype=float)
            else:
                return None
        for fn in (estimate_rr_robust, estimate_rr_short, estimate_rr_lenient):
            rr = fn(green, fs)
            if rr is not None:
                return rr
        return None

    def _br_from_pulse_waveform(self, fs):
        filled = min(self._signal_filled, self.signal_size)
        if filled < min_rr_samples(fs, seconds=3.0):
            return None
        fs = max(fs, 3.0)
        self.pulse.framerate = fs
        rows = self.signal[:filled]
        mask = np.any(rows > 1.0, axis=1)
        if mask.sum() < min_rr_samples(fs, seconds=3.0):
            return None
        sig = rows[mask]
        try:
            wave = moving_avg(self.pulse.get_pulse(sig), 6)
            if len(wave) < min_rr_samples(fs, seconds=2.5):
                return None
            return estimate_rr_from_envelope(wave, fs)
        except (ValueError, TypeError):
            return None

    def _br_from_ml(self, fs):
        if not ml_pulse_available():
            return None, None
        n_frames = max(self.frame_count, self.face_ok_frames)
        rr_wave = self.ml_pulse.estimate_rr(fs, n_frames=n_frames)
        rr_motion = self.ml_pulse.estimate_rr_motion(fs, n_frames=n_frames)
        return rr_wave, rr_motion

    def _estimate_br(self, live_only=False):
        fs = max(self._effective_fps(), 3.0)
        min_n = min_rr_samples(fs, seconds=2.5)
        estimates = []

        rr_ml_wave, rr_ml_motion = self._br_from_ml(fs)
        if rr_ml_wave is not None:
            estimates.extend([rr_ml_wave] * 5)
        if rr_ml_motion is not None:
            estimates.extend([rr_ml_motion] * 4)
        if rr_ml_wave is not None and rr_ml_motion is not None:
            if abs(rr_ml_wave - rr_ml_motion) <= 4:
                estimates.extend([rr_ml_wave] * 3)

        green = self._trim_trace(self.br_green_trace, fs)
        motion = self._trim_trace(self.br_motion_trace, fs)
        face_g = self._trim_trace(self.face_green, fs)

        face_traces = (
            (face_g, 4),
            (green, 3),
            (motion, 2),
        )
        for trace, weight in face_traces:
            if len(trace) >= min_n:
                rr = estimate_rr_robust(trace, fs)
                if rr is not None:
                    estimates.extend([rr] * weight)

        signal_green_rr = self._br_from_signal_green(fs)
        if signal_green_rr is not None:
            estimates.extend([signal_green_rr, signal_green_rr, signal_green_rr])

        pulse_rr = self._br_from_pulse_waveform(fs)
        if pulse_rr is not None:
            estimates.extend([pulse_rr, pulse_rr])

        br = fuse_rr(estimates)
        if br is None:
            for trace in (face_g, green, motion):
                if len(trace) >= min_rr_samples(fs, seconds=2.5):
                    rr = estimate_rr_lenient(trace, fs)
                    if rr is not None:
                        br = round(rr)
                        break
        if br is None:
            rr = self._br_from_signal_green(fs)
            if rr is not None:
                br = round(rr) if isinstance(rr, float) else rr
        return br

    def _hr_from_signal(self, fs):
        rgb = self._valid_signal_rows()
        if rgb is None:
            if len(self.face_green) >= 15:
                g = np.asarray(self.face_green, dtype=float)
                rgb = np.column_stack([g, g, g])
            else:
                return None
        return self._estimate_hr_from_rgb(rgb)

    def _get_pulse_waveform(self, fs):
        n_frames = max(len(self.face_green), self.frame_count)
        if self._scan_accumulated > 2.0 and n_frames > 30:
            fs = n_frames / self._scan_accumulated

        if ml_pulse_available() and self.ml_pulse.waveform_len >= 60 and n_frames >= 60:
            wave = self.ml_pulse.get_waveform_resampled(n_frames)
            if wave is not None and len(wave) >= 60:
                return wave, fs

        rgb = self._valid_signal_rows()
        if rgb is None or len(rgb) < 60:
            if len(self.face_green) >= 60:
                g = np.asarray(self.face_green, dtype=float)
                rgb = np.column_stack([g, g, g])
            else:
                return None, fs
        self.pulse.framerate = fs
        old_sz = self.pulse.signal_size
        try:
            self.pulse.signal_size = len(rgb)
            wave = moving_avg(self.pulse.get_pulse(rgb), 6)
            return wave, fs
        finally:
            self.pulse.signal_size = old_sz

    def _estimate_bp(self, fs, hr=None, br=None, age=None, sex=None):
        if not bp_model_available():
            preload_bp_model()
        wave, use_fs = self._get_pulse_waveform(fs)
        if wave is None:
            return None
        green = self._valid_green_trace()
        return estimate_bp(
            wave, use_fs, hr=hr, breathing_rate=br,
            green_trace=green, age=age, sex=sex,
        )

    def _final_results(self):
        fs = max(self._effective_fps(), 8.0)
        final_hr = None
        median_hr = None
        fallback_hr = None
        hr_confidence = "low"
        hr_spread = None

        full_window_hr = self._estimate_hr_full_window()

        if len(self.hrs) > 0:
            raw_pool = self.hrs[-40:] if len(self.hrs) >= 40 else self.hrs
            pool = [h for h in raw_pool if 58 <= h <= 130]
            if len(pool) < 3:
                pool = [h for h in raw_pool if 52 <= h <= 140]
            if len(pool) < 1:
                pool = list(raw_pool)
            median_hr = round(float(np.median(pool)))
            trimmed = sorted(pool)
            if len(trimmed) > 4:
                trimmed = trimmed[1:-1]
            trimmed_mean = round(float(np.mean(trimmed)))
            fallback_hr = round(float(np.median(pool[-8:]))) if len(pool) > 5 else round(pool[-1])

            if full_window_hr is not None:
                final_hr = full_window_hr
                if pool:
                    hr_spread = int(max(pool) - min(pool)) if len(pool) > 1 else 0
                if pool and abs(full_window_hr - median_hr) <= 8:
                    hr_confidence = "high"
                elif pool and abs(full_window_hr - median_hr) <= 14:
                    hr_confidence = "medium"
                    final_hr = round(float(np.median([full_window_hr, median_hr])))
                else:
                    hr_confidence = "low"
            else:
                candidates = [c for c in (median_hr, trimmed_mean, fallback_hr) if c is not None]
                spread = max(candidates) - min(candidates) if len(candidates) > 1 else 0
                hr_confidence = "high" if spread <= 8 else ("medium" if spread <= 15 else "low")
                final_hr = round(float(np.median(candidates))) if candidates else median_hr
        else:
            final_hr = full_window_hr or self._hr_from_signal(fs)
            if final_hr is not None:
                median_hr = final_hr
                fallback_hr = final_hr

        if final_hr is None and full_window_hr is not None:
            final_hr = full_window_hr
            median_hr = full_window_hr

        if final_hr is None and self.latest_hr is not None:
            final_hr = self.latest_hr
            median_hr = self.latest_hr

        if final_hr is None:
            rgb = self._valid_signal_rows()
            n = len(self.face_green)
            if rgb is None and n >= 20:
                g = np.asarray(self.face_green, dtype=float)
                rgb = np.column_stack([g, g, g])
            if rgb is not None and len(rgb) >= 20:
                use_fs = fs
                if self._scan_accumulated > 1.5 and n >= 20:
                    use_fs = max(n / self._scan_accumulated, 8.0)
                hr_try = self._estimate_hr_from_rgb(rgb, fs_override=use_fs)
                if hr_try is not None:
                    final_hr = hr_try
                    median_hr = hr_try

        final_br = self._estimate_br(live_only=False)
        if final_br is None and ml_pulse_available():
            fs_br = max(self._effective_fps(), 3.0)
            rr_ml, _ = self._br_from_ml(fs_br)
            if rr_ml is not None:
                final_br = rr_ml
        if final_br is None and self.latest_br is not None:
            final_br = self.latest_br
        if final_br is None:
            fs_br = max(self._effective_fps(), 3.0)
            for trace in (self.face_green, self.br_green_trace, self.br_motion_trace):
                if len(trace) >= 25:
                    rr = estimate_rr_lenient(np.asarray(trace, dtype=float), fs_br)
                    if rr is not None:
                        final_br = round(rr)
                        break
            if final_br is None:
                rr = self._br_from_signal_green(fs_br)
                if rr is not None:
                    final_br = round(rr) if isinstance(rr, float) else rr

        face_pct = (self.face_ok_frames / max(self.frame_count, 1)) * 100

        bp_result = self._estimate_bp(fs, hr=final_hr, br=final_br)

        result = {
            "heart_rate": final_hr,
            "heart_rate_median": median_hr if median_hr is not None else final_hr,
            "heart_rate_fallback": fallback_hr if fallback_hr is not None else final_hr,
            "breathing_rate": final_br,
            "live_hr": self.latest_hr,
            "live_br": self.latest_br,
            "duration": self.duration,
            "frames": self.frame_count,
            "signal_samples": int(min(self._signal_filled, self.signal_size)),
            "face_green_samples": len(self.face_green),
            "fps": round(fs, 2),
            "hr_confidence": hr_confidence,
            "hr_spread": hr_spread,
            "face_detected_pct": round(face_pct),
            "success": final_hr is not None or final_br is not None,
        }
        if bp_result:
            result.update(bp_result)
        return result

    def run(self):
        try:
            self.load_model()
        except Exception as exc:
            self.emit("error", message=str(exc))
            return

        if self.remote_mode:
            self.emit("status", message="Center your face in the camera", phase="preview")
            return

        camera = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(self.source)
        if not camera.isOpened():
            self.emit("error", message="Webcam not found. Close other apps using the camera.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.emit("status", message="Center your face in the camera", phase="preview")

        while not self.stop_flag:
            grabbed, orig = camera.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            event = self.process_frame(orig)
            if event and event.get("type") == "complete":
                self.emit("complete", **{k: v for k, v in event.items() if k != "type"})
                break

        camera.release()

        if not self.preview_mode and self.scan_start_time is not None and not self.stop_flag:
            if self._batch_buf:
                pad = self.batch_size - len(self._batch_buf)
                batch = np.stack(self._batch_buf)
                if pad > 0:
                    batch = np.concatenate([batch, np.zeros((pad, *batch.shape[1:]), dtype=batch.dtype)])
                self.process_batch(batch)
            self.emit("complete", **self._final_results())

    def stop(self):
        self.stop_flag = True
