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
from respiration import estimate_rr_robust, fuse_rr
from utils import moving_avg
from paths import MODEL_PATH

torch.set_num_threads(2)

_MODEL_CACHE = {"model": None, "transform": None, "device": None}


class VitalsScanner:
    def __init__(self, duration=30, source=0, on_event=None, remote_mode=False):
        self.duration = duration
        self.source = source
        self.on_event = on_event or (lambda e: None)
        self.remote_mode = remote_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 10
        self.signal_size = 75 if duration >= 45 else 60
        self.pulse_framerate = 25.0  # same nominal FPS as terminal process_mask.py
        self.stop_flag = False

        self.signal = np.zeros((self.signal_size, 3))
        self.pulse = Pulse(self.pulse_framerate, self.signal_size, self.batch_size)
        self.hrs = []
        self.brs = []
        self.chest_green = []
        self.chest_motion = []
        self.face_green = []
        self.face_ok_frames = 0
        self.chest_active_frames = 0
        self.frame_count = 0
        self.actual_fps = 25.0
        self.latest_br = None
        self.latest_hr = None

        self._batch_buf = []
        self._signal_filled = 0
        self.model = None
        self.preview_mode = True
        self.scan_start_time = None
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

    def _effective_fps(self):
        if self.remote_mode and self._client_fps_ready:
            return max(8.0, min(30.0, float(self.actual_fps)))
        if self.scan_start_time and self._scan_fps_start:
            return max(8.0, min(30.0, float(self.actual_fps)))
        return self.pulse_framerate

    def note_client_timestamp(self, ts_ms):
        if self._last_client_ts is not None:
            dt = (float(ts_ms) - self._last_client_ts) / 1000.0
            if 0.028 < dt < 0.22:
                self._client_ts_buf.append(dt)
                if len(self._client_ts_buf) > 40:
                    self._client_ts_buf.pop(0)
                if len(self._client_ts_buf) >= 6:
                    self.actual_fps = len(self._client_ts_buf) / sum(self._client_ts_buf)
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

    def abort_scan(self):
        """User cancelled mid-scan — reset without closing connection."""
        self.finish_scan()
        self._reset_vitals()

    def start_scan(self, duration=None):
        if duration:
            self.duration = duration
            self.signal_size = 75 if duration >= 45 else 60
        self._reset_vitals()
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
        self._last_mask = None
        self._last_pred = None

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

        send_mask = scanning or self._overlay_tick % 2 == 0
        if send_mask:
            crop = overlay_alpha[y1:y2 + 1, x1:x2 + 1]
            if crop.size > 0:
                small = cv2.resize(crop, (96, 120), interpolation=cv2.INTER_LINEAR)
                small = cv2.GaussianBlur(small, (0, 0), 1.2)
                alpha = (np.clip(small, 0, 1) * 255).astype(np.uint8)
                bgra = np.zeros((small.shape[0], small.shape[1], 4), dtype=np.uint8)
                bgra[:, :, 1] = (small * 200).astype(np.uint8)
                bgra[:, :, 2] = (small * 120).astype(np.uint8)
                bgra[:, :, 3] = alpha
                _, buf = cv2.imencode(".png", bgra)
                meta["mask_fmt"] = "png"
                meta["mask"] = base64.b64encode(buf).decode("ascii")

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
        face_pixels = int(face_mask.sum())
        chest_pixels = int(chest_mask.sum())
        face_ratio = face_pixels / (face_mask.size + 1e-6)
        chest_ratio = chest_pixels / (chest_mask.size + 1e-6)
        face_ok = face_pixels > 80 and face_ratio > 0.03

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
        if chest_pixels > 80 and chest_ratio > 0.03:
            chest_bgr = orig[chest_y0:][chest_mask]
            ys, _ = np.where(chest_mask)
            info["active"] = True
            info["green"] = float(chest_bgr[:, 1].mean())
            info["motion"] = float(ys.mean() + chest_y0)
        return info, chest_y0

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

    def process_frame(self, orig):
        """Process one camera frame — used for VPS (browser camera) and local webcam."""
        if self.model is None or self._img_transform is None:
            raise RuntimeError("Model not loaded")

        orig = self._prepare_frame(orig)
        shape = orig.shape[:2]
        scanning = not self.preview_mode and self.scan_start_time is not None

        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), cv2.INTER_LINEAR)
        tensor = self._img_transform(Image.fromarray(resized)).unsqueeze(0)
        imgs = tensor.to(dtype=torch.float, device=self.device)

        with torch.inference_mode():
            pred = self.model(imgs)
        pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
        pred_np = pred.squeeze().cpu().numpy()
        mask_bool = pred_np > 0.8

        frame_full = orig.copy()
        region_info, chest_y0 = self.extract_regions(frame_full, mask_bool, pred_np)
        face_skin = region_info["face_skin"]
        overlay_alpha = region_info["overlay_alpha"]
        lighting_score, lighting_msg = self._assess_lighting(frame_full, face_skin)

        if scanning:
            self._update_scan_fps()
            if region_info.get("face_ok"):
                self.face_ok_frames += 1
            if region_info.get("active"):
                self.chest_active_frames += 1
                self.chest_green.append(region_info["green"])
                self.chest_motion.append(region_info["motion"])

            masked = frame_full.copy()
            masked[mask_bool == 0] = 0
            self._batch_buf.append(masked)
            self.frame_count += 1

            if len(self._batch_buf) >= self.batch_size:
                batch = np.stack(self._batch_buf[: self.batch_size])
                self._batch_buf = self._batch_buf[self.batch_size :]
                self.process_batch(batch)

            elapsed = time.time() - self.scan_start_time
            if elapsed >= self.duration:
                if self._batch_buf:
                    pad = self.batch_size - len(self._batch_buf)
                    batch = np.stack(self._batch_buf)
                    if pad > 0:
                        batch = np.concatenate(
                            [batch, np.zeros((pad, *batch.shape[1:]), dtype=batch.dtype)]
                        )
                    self.process_batch(batch)
                    self._batch_buf = []
                return {"type": "complete", **self._final_results()}
            remaining = max(0, int(self.duration - elapsed))
        else:
            remaining = self.duration

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
            "face_ok": region_info["face_ok"],
            "condition_score": condition,
            "lighting_message": lighting_msg,
            "lighting_ok": lighting_score >= 0.65,
            "progress": progress,
            "overlay": overlay,
            "scan_fps": round(self._effective_fps(), 1) if scanning else None,
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

        if avg_face <= 80 or face_ratio <= 0.03:
            return

        means = np.zeros((self.batch_size, 3))
        for fi in range(self.batch_size):
            if face_pixels[fi] > 0:
                means[fi] = face_batch[fi][face_skin[fi]].mean(axis=0)
                self.face_green.append(float(means[fi, 1]))

        b = self.batch_size
        if self._signal_filled >= self.signal_size:
            self.signal[:-b] = self.signal[b:]
            self.signal[-b:] = means
            self.pulse.framerate = self._effective_fps()
            p = moving_avg(self.pulse.get_pulse(self.signal), 6)
            hr = float(self.pulse.get_rfft_hr(p))
            if not np.isfinite(hr) or hr < 45 or hr > 180:
                return
            if len(self.hrs) >= 4:
                med = float(np.median(self.hrs[-8:]))
                if abs(hr - med) > 14:
                    return
            if len(self.hrs) > 300:
                self.hrs.pop(0)
            self.hrs.append(hr)
            smoothed = moving_avg(self.hrs, 5)[-1] if len(self.hrs) > 4 else hr
            self.latest_hr = round(smoothed)
        else:
            end = min(self._signal_filled + b, self.signal_size)
            self.signal[self._signal_filled:end] = means[: end - self._signal_filled]
        self._signal_filled += b

        br = self._estimate_br()
        if br is not None:
            self.latest_br = br
            self.brs.append(br)

    def _estimate_br(self):
        fs = max(self._effective_fps(), 1.0)
        min_chest = int(fs * 6)
        min_face = int(fs * 8)
        weighted = []

        if len(self.chest_motion) >= min_chest:
            rr = estimate_rr_robust(np.array(self.chest_motion), fs)
            if rr is not None:
                weighted.extend([rr, rr])
        if len(self.chest_green) >= min_chest:
            rr = estimate_rr_robust(np.array(self.chest_green), fs)
            if rr is not None:
                weighted.extend([rr, rr])
        if len(self.face_green) >= min_face:
            rr = estimate_rr_robust(np.array(self.face_green), fs)
            if rr is not None:
                weighted.append(rr)

        br = fuse_rr(weighted)
        if br is not None and self.brs:
            recent = self.brs[-8:] + [br]
            return round(float(np.median(recent)))
        return round(br) if br is not None else None

    def _final_results(self):
        fs = self._effective_fps()
        final_hr = None
        median_hr = None
        fallback_hr = None

        if len(self.hrs) > 0:
            fs = self._effective_fps()
            self.pulse.framerate = fs
            p = moving_avg(self.pulse.get_pulse(self.signal), 6)
            fft_hr = round(float(self.pulse.get_rfft_hr(p)))
            pool = self.hrs[-15:] if len(self.hrs) >= 15 else self.hrs
            median_hr = round(float(np.median(pool)))
            trimmed = sorted(pool)
            if len(trimmed) > 4:
                trimmed = trimmed[1:-1]
            trimmed_mean = round(float(np.mean(trimmed)))
            fallback_hr = round(moving_avg(self.hrs, 8)[-1]) if len(self.hrs) > 7 else round(self.hrs[-1])
            final_hr = median_hr
            candidates = [median_hr, trimmed_mean, fallback_hr]
            if 45 <= fft_hr <= 180:
                candidates.append(fft_hr)
            in_band = [c for c in candidates if 50 <= c <= 120]
            if in_band:
                final_hr = round(float(np.median(in_band)))
            elif abs(fft_hr - median_hr) <= 10:
                final_hr = fft_hr
        elif len(self.face_green) >= fs * 15 and np.any(self.signal):
            self.pulse.framerate = fs
            n = min(len(self.face_green), self.signal_size)
            p = moving_avg(self.pulse.get_pulse(self.signal[:n]), 6)
            final_hr = round(self.pulse.get_rfft_hr(p))
            median_hr = final_hr
            fallback_hr = final_hr

        final_br = self._estimate_br()
        if final_br is None and self.brs:
            final_br = fuse_rr(self.brs[-8:])

        face_pct = (self.face_ok_frames / max(self.frame_count, 1)) * 100
        chest_pct = (self.chest_active_frames / max(self.frame_count, 1)) * 100

        return {
            "heart_rate": final_hr,
            "heart_rate_median": median_hr,
            "heart_rate_fallback": fallback_hr,
            "breathing_rate": final_br,
            "duration": self.duration,
            "frames": self.frame_count,
            "fps": round(fs, 2),
            "face_detected_pct": round(face_pct),
            "chest_detected_pct": round(chest_pct),
        }

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
