import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pybaselines import Baseline
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

# ───────────────── 1. Data Processing (Sanitized) ──────────────────── #

def airpls_baseline(intensities: np.ndarray, lam: float = 1e5) -> np.ndarray:
    """Baseline subtraction using AirPLS (compatible across versions)."""
    bl = Baseline()
    y = intensities.astype(np.float32)
    for kwargs in [dict(lam=lam, itermax=50), dict(lam=lam)]:
        try:
            baseline, _ = bl.airpls(y, **kwargs)
            return y - baseline.astype(np.float32)
        except: continue
    return y

def _to_phi(series: np.ndarray) -> np.ndarray:
    """Map normalized series to angular space."""
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
    dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    return dist

class RamanDataset(Dataset):
    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def prepare_data(csv_path, start_col, end_col):
    """Loads CSV, performs baseline correction, and generates 3-channel images."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    y = df['label'].values.astype(np.float32)
    # Extract spectral columns based on indices
    X_raw = df.drop(columns=['label']).iloc[:, start_col:end_col].values.astype(np.float32)

    n_samples, n_pts = X_raw.shape
    X_imgs = np.empty((n_samples, 3, n_pts, n_pts), dtype=np.float32)
    
    for i in tqdm(range(n_samples), desc="Preprocessing"):
        s = airpls_baseline(X_raw[i])
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        # Channel 0: GASF | Channel 1: GADF | Channel 2: Recurrence Plot
        X_imgs[i, 0], X_imgs[i, 1], X_imgs[i, 2] = gasf(s), gadf(s), recurrence_plot(s)
    return X_imgs, y

# ───────────────── 2. Model & Interpretability (Grad-CAM) ────────────────── #

class GradCAM:
    """Grad-CAM implementation optimized for Regression tasks."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Hooks to capture gradients and feature maps
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _save_activation(self, module, input, output):
        self.activations = output

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        # In regression, we compute gradients directly w.r.t the scalar output
        output.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # Apply ReLU and normalize
        cam = np.maximum(cam.cpu().numpy(), 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam

def build_model(backbone='resnet50'):
    """Initializes ResNet and adapts input/output for spectral regression."""
    model = getattr(models, backbone)(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# ───────────────── 3. Training & Visualization ────────────────── #

def run_training(args):
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
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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
        print(f"Epoch {epoch} | R2: {r2:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"--> Saved new best model (R2: {best_r2:.4f})")

    # Final Visualization
    visualize_results(model, test_loader, device)

def visualize_results(model, loader, device, output_dir="results"):
    """Generates and saves Grad-CAM heatmaps for sample test data."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # Targeting the last convolutional layer of ResNet
    target_layer = model.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    imgs, labels = next(iter(loader))
    print(f"\nGenerating Grad-CAM visualizations in '{output_dir}'...")
    
    for i in range(min(5, len(labels))):
        input_tensor = imgs[i:i+1].to(device)
        heatmap = cam_tool.generate_cam(input_tensor)
        
        # Displaying with Channel 0 (GASF) as background
        raw_img = imgs[i, 0].numpy()
        raw_img = np.uint8(255 * (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min()))
        
        heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Overlay heatmap on raw image
        background = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(background, 0.5, heatmap_color, 0.5, 0)
        
        cv2.imwrite(os.path.join(output_dir, f"sample_{i}_cam.jpg"), overlay)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raman Spectrum ResNet Regression')
    parser.add_argument('--csv_path', type=str, default='data.csv', help='Path to spectral CSV file')
    parser.add_argument('--start_idx', type=int, default=30, help='Start index of spectral columns')
    parser.add_argument('--end_idx', type=int, default=530, help='End index of spectral columns')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    run_training(args)