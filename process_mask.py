import sys
import time
from threading import Thread

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pulse import Pulse
from respiration import estimate_rr_robust, fuse_rr
from utils import moving_avg
from paths import BRS_PATH, FFT_SPEC_PATH, HRS_PATH, RESULTS_PNG_PATH


class ProcessMasks():

    def __init__(self, sz=270, fs=30, bs=30, size=256, duration=60):
        print('init')
        self.stop = False
        self.vitals_done = False
        self.duration = duration
        self.masked_batches = []
        self.batch_mean = []
        self.signal_size = sz
        self.batch_size = bs
        self.signal = np.zeros((sz, 3))
        self.pulse = Pulse(fs, sz, bs, size)
        self.actual_fps = float(fs)
        self.hrs = []
        self.brs = []
        self.face_green = []
        self.chest_green = []
        self.chest_motion = []
        self.face_ok_frames = 0
        self.chest_active_frames = 0
        self.frame_count = 0
        self.latest_hr = None
        self.latest_br = None
        self.save_results = True

    def __call__(self, pipe, plot_pipe, source):
        self.pipe = pipe
        self.plot_pipe = plot_pipe
        self.source = source
        compute_mean_thread = Thread(target=self.compute_mean)
        compute_mean_thread.start()

        extract_signal_thread = Thread(target=self.extract_signal)
        extract_signal_thread.start()

        self.rec_frames()

        compute_mean_thread.join()
        extract_signal_thread.join()

    def rec_frames(self):
        batch_buf = []
        while True and not self.stop:
            data = self.pipe.recv()

            if data is None:
                if batch_buf:
                    self._push_batch(batch_buf)
                self.terminate()
                break

            if isinstance(data, tuple):
                if len(data) >= 2 and data[0] == 'fps':
                    self.actual_fps = float(data[1])
                    if len(data) > 2:
                        self.frame_count = int(data[2])
                    continue
                if data[0] == 'vitals_done':
                    self.vitals_done = True
                    if batch_buf:
                        self._push_batch(batch_buf)
                        batch_buf = []
                    continue

            masked, chest_info = data
            self.frame_count += 1

            if chest_info.get('face_ok'):
                self.face_ok_frames += 1
            if chest_info.get('active'):
                self.chest_active_frames += 1
                self.chest_green.append(chest_info['green'])
                self.chest_motion.append(chest_info['motion'])

            batch_buf.append((masked, chest_info))
            if len(batch_buf) >= self.batch_size:
                self._push_batch(batch_buf)
                batch_buf = []

    def _push_batch(self, batch_buf):
        masks = np.stack([m for m, _ in batch_buf])
        self.masked_batches.append(masks)

    def _estimate_br(self):
        fs = self.actual_fps
        estimates = []
        if len(self.chest_motion) >= fs * 12:
            estimates.append(estimate_rr_robust(np.array(self.chest_motion), fs))
        if len(self.chest_green) >= fs * 12:
            estimates.append(estimate_rr_robust(np.array(self.chest_green), fs))
        if len(self.face_green) >= fs * 12:
            estimates.append(estimate_rr_robust(np.array(self.face_green), fs))
        return fuse_rr(estimates)

    def process_signal(self, batch_mean):
        size = self.signal.shape[0]
        b_size = batch_mean.shape[0]

        self.signal[0:size - b_size] = self.signal[b_size:size]
        self.signal[size - b_size:] = batch_mean
        self.pulse.framerate = self.actual_fps
        p = self.pulse.get_pulse(self.signal)
        p = moving_avg(p, 6)
        hr = self.pulse.get_rfft_hr(p)
        if len(self.hrs) > 300:
            self.hrs.pop(0)

        self.hrs.append(hr)
        self.latest_hr = round(moving_avg(self.hrs, 3)[-1]) if len(self.hrs) > 2 else round(hr)

        br = self._estimate_br()
        if br is not None:
            self.latest_br = br
            self.brs.append(br)

        if self.plot_pipe is not None and self.stop:
            self.plot_pipe.send(None)
        elif self.plot_pipe is not None:
            self.plot_pipe.send([p, self.hrs])
        else:
            hr_txt = self.latest_hr if self.latest_hr is not None else '--'
            br_txt = self.latest_br if self.latest_br is not None else '--'
            sys.stdout.write(f'\rHR: {hr_txt}  BR: {br_txt}  ')
            sys.stdout.flush()

    def extract_signal(self):
        signal_extracted = 0

        while True and not self.stop:
            if len(self.batch_mean) == 0:
                time.sleep(0.01)
                continue

            mean_dict = self.batch_mean.pop(0)
            mean = mean_dict['mean']

            if mean_dict['face_detected'] is False:
                if self.plot_pipe is not None:
                    self.plot_pipe.send('no face detected')
                continue

            for fi in range(mean.shape[0]):
                if np.any(mean[fi]):
                    self.face_green.append(float(mean[fi, 1]))

            if signal_extracted >= self.signal_size:
                self.process_signal(mean)
            else:
                end = min(signal_extracted + mean.shape[0], self.signal_size)
                self.signal[signal_extracted:end] = mean[:end - signal_extracted]
            signal_extracted += mean.shape[0]

    def compute_mean(self):
        while True and not self.stop:
            if len(self.masked_batches) == 0:
                time.sleep(0.01)
                continue

            batch = self.masked_batches.pop(0)
            h, w = batch.shape[1], batch.shape[2]
            chest_y0 = int(h * 0.45)
            face_batch = batch[:, :chest_y0, :, :]
            face_area = chest_y0 * w
            face_skin = (face_batch != 0).any(axis=3)
            face_pixels = face_skin.sum(axis=(1, 2))
            avg_face = face_pixels.mean()
            face_ratio = avg_face / (face_area + 1e-6)

            m = {'face_detected': False, 'mean': np.zeros((self.batch_size, 3))}
            if avg_face > 80 and face_ratio > 0.03:
                m['face_detected'] = True
                for fi in range(self.batch_size):
                    if face_pixels[fi] > 0:
                        m['mean'][fi] = face_batch[fi][face_skin[fi]].mean(axis=0)

            self.batch_mean.append(m)

    def print_final_result(self):
        fs = self.actual_fps
        final_hr = None
        median_hr = None
        fallback_hr = None

        if len(self.hrs) > 0:
            self.pulse.framerate = fs
            p = moving_avg(self.pulse.get_pulse(self.signal), 6)
            final_hr = round(self.pulse.get_rfft_hr(p))
            median_hr = round(float(np.median(self.hrs[-10:] if len(self.hrs) >= 10 else self.hrs)))
            fallback_hr = round(moving_avg(self.hrs, 6)[-1]) if len(self.hrs) > 5 else round(self.hrs[-1])
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

        print('\n' + '=' * 50)
        print('VITALS SCAN RESULTS')
        print('=' * 50)
        print(f'Duration:     {self.duration}s')
        print(f'Frames:       {self.frame_count}')
        print(f'FPS:          {fs:.2f}')
        print(f'Face OK:      {face_pct:.0f}%')
        print(f'Chest OK:     {chest_pct:.0f}%')
        print('-' * 50)
        print(f'Heart Rate:   {final_hr if final_hr is not None else "N/A"} BPM')
        if median_hr is not None:
            print(f'HR (median):  {median_hr} BPM')
        if fallback_hr is not None and fallback_hr != final_hr:
            print(f'HR (fallback): {fallback_hr} BPM')
        print(f'Breathing:    {final_br if final_br is not None else "N/A"} breaths/min')
        print('=' * 50)

        return {
            'heart_rate': final_hr,
            'heart_rate_median': median_hr,
            'heart_rate_fallback': fallback_hr,
            'breathing_rate': final_br,
            'duration': self.duration,
            'frames': self.frame_count,
            'fps': round(fs, 2),
            'face_detected_pct': round(face_pct),
            'chest_detected_pct': round(chest_pct),
        }

    def terminate(self):
        if self.plot_pipe is not None:
            self.plot_pipe.send(None)
        self.print_final_result()
        self.savePlot(self.source)
        self.saveresults()
        self.stop = True

    def saveresults(self):
        """Save heart rates and power spectrum arrays."""
        np.save(HRS_PATH, np.array(self.hrs))
        np.save(FFT_SPEC_PATH, np.array(self.pulse.fft_spec))
        if self.brs:
            np.save(BRS_PATH, np.array(self.brs))

    def savePlot(self, path):
        if self.save_results is False:
            return

        if len(self.hrs) == 0:
            return

        ax1 = plt.subplot(1, 1, 1)
        ax1.set_title('HR')
        ax1.set_ylim([20, 180])
        ax1.plot(moving_avg(self.hrs, 6))

        plt.tight_layout()
        plt.savefig(RESULTS_PNG_PATH)
        plt.close()
