"""
belief_estimator_train.py — E3: Train sliding-window MLP regime classifier.

Reads trajectories from results/e0_oracle_heuristic/trajectories/ (produced by
run_oracle_heuristic.py --save-trajectories).  Each file contains:
  {"obs": [[7 floats] × T steps], "regime": "val" | "mom", "seed": int}

Creates overlapping windows of W=20 consecutive steps per trajectory, labels each
window with the episode regime, and trains a 2-class MLP:
  Input:  W × 7 = 140 features (zero-padded at episode start)
  Hidden: [128, 64] with ReLU
  Output: 2 logits (val=0, mom=1)
  Loss:   CrossEntropyLoss
  Opt:    Adam lr=1e-3, 50 epochs, 80/20 train/val split

Saves model weights to results/e3_belief_estimator/estimator.pt
Also writes training_log.json with per-epoch loss/accuracy.

Usage:
  .venv/bin/python abides-gym/scripts/belief_estimator_train.py
  .venv/bin/python abides-gym/scripts/belief_estimator_train.py --traj-dir results/e0_oracle_heuristic/trajectories
  .venv/bin/python abides-gym/scripts/belief_estimator_train.py --epochs 5  # smoke
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REGIME_LABELS = {"val": 0, "mom": 1}
WINDOW = 20
INPUT_DIM = WINDOW * 7  # 140


class RegimeMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: List[int] = None, n_classes: int = 2):
        super().__init__()
        if hidden is None:
            hidden = [128, 64]
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, window_flat: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for a single flattened window (140,)."""
        with torch.no_grad():
            logits = self(torch.as_tensor(window_flat, dtype=torch.float32).unsqueeze(0))
            return torch.softmax(logits, dim=-1).squeeze(0).numpy()


def load_trajectories(traj_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all trajectory files and return (X, y) arrays."""
    X_list, y_list = [], []
    files = sorted(traj_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in {traj_dir}")

    for f in files:
        data = json.loads(f.read_text())
        regime = data.get("regime", "")
        if regime not in REGIME_LABELS:
            continue
        label = REGIME_LABELS[regime]
        obs_seq = np.asarray(data["obs"], dtype=np.float32)  # (T, 7)
        T = len(obs_seq)

        for t in range(T):
            start = max(0, t - WINDOW + 1)
            window = obs_seq[start : t + 1]  # (≤W, 7)
            if len(window) < WINDOW:
                pad = np.zeros((WINDOW - len(window), 7), dtype=np.float32)
                window = np.vstack([pad, window])
            X_list.append(window.reshape(-1))  # (140,)
            y_list.append(label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def train(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.traj_dir)

    print(f"[belief_estimator] Loading trajectories from {traj_dir}", flush=True)
    t0 = time.time()
    X, y = load_trajectories(traj_dir)
    n_samples = len(X)
    val_count = int(n_samples * (1 - args.train_split))
    print(f"[belief_estimator] {n_samples} windows loaded in {time.time()-t0:.1f}s  "
          f"(val={REGIME_LABELS['val']}: {(y==0).sum()}, mom={REGIME_LABELS['mom']}: {(y==1).sum()})",
          flush=True)

    # Shuffle and split.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_samples)
    val_idx = idx[:val_count]
    train_idx = idx[val_count:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx],  y[val_idx]

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                           batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=args.batch_size)

    model = RegimeMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    log = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        correct = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_loss /= len(X_tr)
        tr_acc = correct / len(X_tr)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": val_loss, "val_acc": val_acc})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch={epoch:>3}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}", flush=True)

    # Save best model.
    model.load_state_dict(best_state)
    model_path = out_dir / "estimator.pt"
    torch.save({"model_state_dict": best_state,
                "input_dim": INPUT_DIM,
                "hidden": [128, 64],
                "n_classes": 2,
                "window": WINDOW,
                "best_val_acc": best_val_acc,
                "regime_labels": REGIME_LABELS}, str(model_path))

    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n[belief_estimator] Best val acc: {best_val_acc:.3f}")
    print(f"[belief_estimator] Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="E3: Train regime belief estimator")
    parser.add_argument("--traj-dir", type=str, default="results/e0_oracle_heuristic/trajectories")
    parser.add_argument("--out-dir",  type=str, default="results/e3_belief_estimator")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed",     type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
