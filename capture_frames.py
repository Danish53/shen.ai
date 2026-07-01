import cv2
import numpy as np
import torch
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import time
import sys
from paths import MODEL_PATH


class CaptureFrames():

    def __init__(self, bs, source, show_mask=False, duration=60):
        self.frame_counter = 0
        self.batch_size = bs
        self.duration = duration
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.show_mask = show_mask

    def load_model(self):
        print('Loading AI model... (15-30 sec on CPU, please wait)')
        sys.stdout.flush()
        self.model = LinkNet34(pretrained=False)
        self.model.load_state_dict(
            torch.load(str(MODEL_PATH), weights_only=False, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        print('Model loaded. Starting vitals scan (HR + Breathing)...')
        sys.stdout.flush()

    def extract_regions(self, orig, mask_bool):
        h, w = orig.shape[:2]
        chest_y0 = int(h * 0.45)
        face_mask = mask_bool[:chest_y0, :]
        chest_mask = mask_bool[chest_y0:, :]
        face_pixels = int(face_mask.sum())
        chest_pixels = int(chest_mask.sum())
        face_ratio = face_pixels / (face_mask.size + 1e-6)
        chest_ratio = chest_pixels / (chest_mask.size + 1e-6)
        face_ok = face_pixels > 80 and face_ratio > 0.03
        chest_info = {
            'active': False, 'green': 0.0, 'motion': 0.0, 'ratio': chest_ratio,
            'face_ok': face_ok, 'face_ratio': face_ratio,
        }
        if chest_pixels > 80 and chest_ratio > 0.03:
            chest_bgr = orig[chest_y0:][chest_mask]
            ys, _ = np.where(chest_mask)
            chest_info['active'] = True
            chest_info['green'] = float(chest_bgr[:, 1].mean())
            chest_info['motion'] = float(ys.mean() + chest_y0)
        return chest_info, chest_y0

    def draw_overlay(self, display, remaining, chest_y0, mask_bool, face_ok, chest_ok):
        h, w = display.shape[:2]
        display[mask_bool] = (
            display[mask_bool].astype(np.float32) * 0.72 +
            np.array([0, 210, 0], dtype=np.float32) * 0.28
        ).astype(np.uint8)
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(display, contours, -1, (0, 255, 120), 1)
        face_color = (0, 255, 0) if face_ok else (0, 80, 255)
        cv2.rectangle(display, (8, 8), (w - 8, chest_y0 - 8), face_color, 2)
        cv2.putText(display, 'FACE: OK' if face_ok else 'FACE: show face here', (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_color, 2)
        chest_color = (0, 255, 0) if chest_ok else (0, 165, 255)
        cv2.rectangle(display, (8, chest_y0 + 8), (w - 8, h - 8), chest_color, 2)
        cv2.putText(display, 'CHEST: OK' if chest_ok else 'CHEST: show chest in lower area',
                    (16, chest_y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, chest_color, 2)
        cv2.line(display, (0, chest_y0), (w, chest_y0), (0, 220, 255), 1)
        cv2.putText(display, f'Time: {remaining}s', (w - 160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)

    def capture_frames(self, source):
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(source)
        if not camera.isOpened():
            print('ERROR: Webcam not found. Close other apps using the camera and try again.')
            sys.exit(1)

        print('Webcam opened. Loading model...')
        sys.stdout.flush()
        time.sleep(0.5)
        grabbed, frame = camera.read()
        if grabbed and self.show_mask:
            preview = frame.copy()
            cv2.putText(preview, 'Loading model...', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Vitals Scan', preview)
            cv2.waitKey(1)

        self.load_model()
        self.capture_start = time.time()
        self.frames_count = 0
        print(f'Scanning for {self.duration} seconds...')
        sys.stdout.flush()

        while True:
            elapsed = time.time() - self.capture_start
            if elapsed >= self.duration:
                print(f'\n{self.duration} seconds complete. Stopping camera...')
                actual_fps = self.frames_count / elapsed if elapsed > 0 else 1.0
                self.terminate(camera, actual_fps)
                break

            grabbed, orig = camera.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            shape = orig.shape[0:2]
            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                actual_fps = self.frames_count / elapsed if elapsed > 0 else 1.0
                self.terminate(camera, actual_fps)
                break

            a = img_transform(Image.fromarray(frame))
            imgs = Variable(a.unsqueeze(0).to(dtype=torch.float, device=self.device))
            with torch.no_grad():
                pred = self.model(imgs)
            pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
            mask_bool = pred.squeeze().cpu().numpy() > 0.8

            frame_full = orig.copy()
            region_info, chest_y0 = self.extract_regions(frame_full, mask_bool)
            masked = frame_full.copy()
            masked[mask_bool == 0] = 0
            self.pipe.send([masked, region_info])

            if self.show_mask:
                display = frame_full.copy()
                remaining = max(0, int(self.duration - elapsed))
                self.draw_overlay(display, remaining, chest_y0, mask_bool,
                                  region_info['face_ok'], region_info['active'])
                cv2.imshow('Vitals Scan', display)

            self.frames_count += 1

    def terminate(self, camera, actual_fps=0):
        if actual_fps > 0:
            self.pipe.send(('fps', actual_fps, self.frames_count))
        self.pipe.send(('vitals_done',))
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
