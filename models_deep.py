from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

QUANTILES = (0.1, 0.5, 0.9)

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pinball_loss_torch(y: torch.Tensor, yhat: torch.Tensor, q: float) -> torch.Tensor:
    # y, yhat: (batch,)
    diff = y - yhat
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))

def quantile_loss(y: torch.Tensor, yhat_q: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    # y: (batch,), yhat_q: (batch,3)
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(pinball_loss_torch(y, yhat_q[:, i], float(q)))
    return torch.stack(losses).sum()

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 6
    patience: int = 2
    clip_grad: float = 1.0
    device: str = "cpu"

class QuantileHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class NBeatsLite(nn.Module):
    """A lightweight N-BEATS-style MLP for one-step forecasting with quantile head.

    Note: This is a compact implementation suitable for small datasets and fast reproduction.
    """
    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = QuantileHead(hidden)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, seq_len, feat_dim) -> flatten
        x = x_seq.reshape(x_seq.size(0), -1)
        h = self.mlp(x)
        return self.head(h)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]

class TransformerEncoderModel(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = QuantileHead(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x_seq)
        x = self.pos(x)
        h = self.encoder(x)  # (batch, seq_len, d_model)
        h_last = h[:, -1, :]
        return self.head(h_last)

class InformerLiteModel(nn.Module):
    """Informer-lite proxy: a shallower Transformer encoder with smaller d_model."""
    def __init__(self, feat_dim: int, d_model: int = 48, nhead: int = 3, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = QuantileHead(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x_seq)
        x = self.pos(x)
        h = self.encoder(x)
        h_last = h[:, -1, :]
        return self.head(h_last)

class TFTLite(nn.Module):
    """A compact TFT-inspired model with:
    - per-timestep feature gating (softmax over features)
    - LSTM encoder
    - self-attention
    - quantile head

    Exposes mean feature-selection weights for interpretability.
    """
    def __init__(self, feat_dim: int, d_model: int = 64, lstm_hidden: int = 64, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.gate = nn.Linear(feat_dim, feat_dim)
        self.embed = nn.Linear(feat_dim, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden, num_heads=nhead, dropout=dropout, batch_first=True)
        self.head = QuantileHead(lstm_hidden)

    def forward(self, x_seq: torch.Tensor, return_weights: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x_seq: (batch, seq_len, feat_dim)
        w = torch.softmax(self.gate(x_seq), dim=-1)  # (batch, seq_len, feat_dim)
        xg = w * x_seq
        x = self.embed(xg)
        h, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)
        h_attn, _ = self.attn(h, h, h)
        out = self.head(h_attn[:, -1, :])
        if return_weights:
            return out, w
        return out

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> nn.Module:
    device = torch.device(cfg.device)
    model = model.to(device)

    ds_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    ds_val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = cfg.patience

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = quantile_loss(yb, yhat)
            loss.backward()
            if cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()

        # validation
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)
                losses.append(float(quantile_loss(yb, yhat).cpu().item()))
        val_loss = float(np.mean(losses)) if losses else float("inf")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for (xb,) in dl:
        xb = xb.to(dev)
        yhat = model(xb)
        outs.append(yhat.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

@torch.no_grad()
def tft_feature_importance(model: TFTLite, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    """Mean feature selection weights across all samples and time steps."""
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    acc = None
    n = 0
    for (xb,) in dl:
        xb = xb.to(dev)
        _, w = model(xb, return_weights=True)  # (batch, seq_len, feat_dim)
        w_mean = w.mean(dim=1)  # average over time -> (batch, feat_dim)
        w_sum = w_mean.sum(dim=0)  # (feat_dim,)
        if acc is None:
            acc = w_sum.detach().cpu().numpy()
        else:
            acc += w_sum.detach().cpu().numpy()
        n += xb.size(0)
    assert acc is not None
    return acc / max(n, 1)
