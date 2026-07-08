"""
Train dual BP models: morphology MLP + waveform CNN (BPWaveNet).
Predicts residuals from population baseline 118/76 mmHg (typical resting adult).
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bp_features import extract_bp_features
from bp_model import BPRegressor
from bp_wavenet import BPWaveNet, normalize_waveform, WAVE_LEN
from paths import BP_MODEL_PATH, BP_NORM_PATH, BP_WAVENET_PATH, WEIGHTS_DIR

BASE_SBP = 118.0
BASE_DBP = 76.0


def _synthetic_ppg(fs, seconds, hr, sbp, dbp, stiffness, noise=0.03):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    hr_hz = hr / 60.0
    pp = max(sbp - dbp, 20)
    base = dbp * 0.15 + pp * 0.1
    wave = np.zeros(n, dtype=float)
    for i in range(n):
        cycle = (t[i] * hr_hz) % 1.0
        if cycle < 0.15:
            wave[i] = base + pp * 0.9 * (cycle / 0.15) ** 0.65
        elif cycle < 0.32:
            decay = 1.0 - 0.3 * stiffness * (cycle - 0.15) / 0.17
            wave[i] = base + pp * decay
        else:
            notch = 0.15 * stiffness * math.sin(2 * math.pi * (cycle - 0.32) / 0.68)
            wave[i] = base + pp * (0.45 + notch)
        wave[i] += np.random.normal(0, pp * noise)
    return wave


def generate_dataset(n_samples=3500, fs=25.0, seconds=16.0):
    xs_feat, xs_wave, ys = [], [], []
    for i in range(n_samples):
        if i and i % 1000 == 0:
            print(f"generated {i}/{n_samples}...")
        hr = np.random.normal(78, 12)
        hr = float(np.clip(hr, 58, 108))
        sbp = float(np.clip(np.random.normal(122, 14), 100, 155))
        dbp = float(np.clip(sbp - np.random.uniform(32, 52), 62, 95))
        stiffness = np.clip((sbp - 110) / 50.0 + np.random.normal(0, 0.1), 0.15, 1.6)
        age = np.random.uniform(22, 68)
        sex = np.random.choice(["M", "F"])
        br = np.random.uniform(11, 20)

        wave = _synthetic_ppg(fs, seconds, hr, sbp, dbp, stiffness)
        green = wave * 0.02 + 128 + np.random.normal(0, 1.5, len(wave))

        feat, q = extract_bp_features(
            wave, fs, hr=hr, breathing_rate=br, green_trace=green, age=age, sex=sex,
        )
        wnorm = normalize_waveform(wave, WAVE_LEN)
        if feat is None or wnorm is None or q < 0.25:
            continue

        xs_feat.append(feat)
        xs_wave.append(wnorm.numpy().reshape(-1))
        ys.append([sbp - BASE_SBP, dbp - BASE_DBP])

    return (
        np.asarray(xs_feat, dtype=np.float32),
        np.asarray(xs_wave, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def train_and_save(epochs=35, batch_size=64):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    x_feat, x_wave, y_res = generate_dataset()
    if len(x_feat) < 800:
        raise RuntimeError("Insufficient training samples")

    sbp_std = float(np.std(y_res[:, 0]) + 1e-6)
    dbp_std = float(np.std(y_res[:, 1]) + 1e-6)
    y_norm = np.column_stack([y_res[:, 0] / sbp_std, y_res[:, 1] / dbp_std]).astype(np.float32)

    with open(BP_NORM_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "residual_mode": True,
            "base_sbp": BASE_SBP,
            "base_dbp": BASE_DBP,
            "sbp_mean": BASE_SBP,
            "dbp_mean": BASE_DBP,
            "sbp_std": sbp_std,
            "dbp_std": dbp_std,
        }, f)

    split = int(len(x_feat) * 0.85)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mlp = BPRegressor().to(device)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    ds_mlp = TensorDataset(torch.from_numpy(x_feat[:split]), torch.from_numpy(y_norm[:split]))
    loader_mlp = DataLoader(ds_mlp, batch_size=batch_size, shuffle=True)

    cnn = BPWaveNet().to(device)
    opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    ds_cnn = TensorDataset(
        torch.from_numpy(x_wave[:split]).unsqueeze(1),
        torch.from_numpy(y_norm[:split]),
    )
    loader_cnn = DataLoader(ds_cnn, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        mlp.train()
        cnn.train()
        lm, lc = 0.0, 0.0
        for xb, yb in loader_mlp:
            xb, yb = xb.to(device), yb.to(device)
            opt_mlp.zero_grad()
            loss = loss_fn(mlp(xb), yb)
            loss.backward()
            opt_mlp.step()
            lm += float(loss.item())
        for xb, yb in loader_cnn:
            xb, yb = xb.to(device), yb.to(device)
            opt_cnn.zero_grad()
            loss = loss_fn(cnn(xb), yb)
            loss.backward()
            opt_cnn.step()
            lc += float(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs} mlp={lm/len(loader_mlp):.4f} cnn={lc/len(loader_cnn):.4f}")

    mlp.eval()
    cnn.eval()
    with torch.no_grad():
        pm = mlp(torch.from_numpy(x_feat[split:]).to(device)).cpu().numpy()
        pc = cnn(torch.from_numpy(x_wave[split:]).unsqueeze(1).to(device)).cpu().numpy()
        for name, pred in [("MLP", pm), ("CNN", pc)]:
            denorm = np.column_stack([
                BASE_SBP + pred[:, 0] * sbp_std,
                BASE_DBP + pred[:, 1] * dbp_std,
            ])
            mae_s = float(np.mean(np.abs(denorm[:, 0] - (y_res[split:, 0] + BASE_SBP))))
            mae_d = float(np.mean(np.abs(denorm[:, 1] - (y_res[split:, 1] + BASE_DBP))))
            print(f"{name} holdout MAE SBP={mae_s:.1f} DBP={mae_d:.1f} mmHg")

    torch.save(mlp.state_dict(), str(BP_MODEL_PATH))
    torch.save(cnn.state_dict(), str(BP_WAVENET_PATH))
    print(f"saved {BP_MODEL_PATH} and {BP_WAVENET_PATH}")


if __name__ == "__main__":
    train_and_save()
