"""
Raman Spectrum Regression with ResNet (3-Channel GASF + GADF + Euclidean RP)
Transform 1D Raman spectra into 2D multi-channel images for deep learning regression.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pybaselines import Baseline
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# ----------------- Preprocessing & Transforms ----------------- #

def airpls_baseline(intensities: np.ndarray, lam: float = 1e5) -> np.ndarray:
    """Baseline subtraction using AirPLS (compatible across pybaselines versions)."""
    bl = Baseline()
    y = intensities.astype(np.float32)
    # Robust parameter handling for different version signatures
    params = [
        {'lam': lam, 'porder': 1, 'itermax': 50},
        {'lam': lam, 'ratio': 0.01, 'itermax': 50},
        {'lam': lam, 'itermax': 50}
    ]
    for p in params:
        try:
            baseline, _ = bl.airpls(y, **p)
            return y - baseline.astype(np.float32)
        except (TypeError, ValueError):
            continue
    return y # Fallback

def _to_phi(series: np.ndarray) -> np.ndarray:
    """Map normalized series to angular space [0, pi]."""
    # Scale to [-1, 1]
    scaled = 2 * (series - series.min()) / (series.max() - series.min() + 1e-9) - 1.0
    return np.arccos(np.clip(scaled, -1.0, 1.0))

def gasf(series: np.ndarray) -> np.ndarray:
    phi = _to_phi(series)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)

def gadf(series: np.ndarray) -> np.ndarray:
    phi = _to_phi(series)
    return np.sin(phi[:, None] - phi[None, :]).astype(np.float32)

def recurrence_plot(series: np.ndarray) -> np.ndarray:
    dist = np.abs(series[:, None] - series[None, :]).astype(np.float32)
    dist -= dist.min()
    dist /= (dist.max() + 1e-9)
    return dist

# ----------------- Dataset & Model ----------------- #

class RamanDataset(Dataset):
    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.X = torch.from_numpy(images)
        self.y = torch.from_numpy(targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_model(backbone: str = 'resnet50'):
    model_fn = getattr(models, backbone)
    model = model_fn(weights=None)
    
    # Adapt first conv layer for 3-channel input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adapt final layer for regression (1 output unit)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ----------------- Main Pipeline ----------------- #

def prepare_data(csv_path: str, start_col: int, end_col: int):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
        
    y = df['label'].values.astype(np.float32)
    # Extract spectral data based on column indices
    X_raw = df.drop(columns=['label']).iloc[:, start_col:end_col].values.astype(np.float32)

    print(f"Processing {len(X_raw)} samples...")
    n_pts = X_raw.shape[1]
    X_imgs = np.empty((len(X_raw), 3, n_pts, n_pts), dtype=np.float32)

    for i in tqdm(range(len(X_raw)), desc="Preprocessing"):
        # 1. Baseline Correction
        spec = airpls_baseline(X_raw[i])
        # 2. Normalization
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
        # 3. 2D Imaging
        X_imgs[i, 0] = gasf(spec)
        X_imgs[i, 1] = gadf(spec)
        X_imgs[i, 2] = recurrence_plot(spec)

    return X_imgs, y

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = prepare_data(args.csv_path, args.start_idx, args.end_idx)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(RamanDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(RamanDataset(X_test, y_test), batch_size=args.batch_size)

    model = build_model(args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_r2 = -np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                out = model(imgs.to(device))
                preds.extend(out.cpu().numpy().flatten())
                trues.extend(targets.numpy().flatten())
        
        r2 = r2_score(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        print(f"Epoch {epoch} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_model.pt')
            pd.DataFrame({'true': trues, 'pred': preds}).to_csv('best_predictions.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='data.csv', help='Path to spectral data')
    parser.add_argument('--start_idx', type=int, default=30, help='Spectral start column index')
    parser.add_argument('--end_idx', type=int, default=530, help='Spectral end column index')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    train(parser.parse_args())