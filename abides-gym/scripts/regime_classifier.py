"""
regime_classifier.py — Learned regime belief estimator for v5 oracle experiments
==================================================================================

Trains two classifiers to output (p_val, p_mom) from observable market history,
replacing the Kim filter (which failed to distinguish Val/Mom regimes) and the
oracle (which requires ground-truth agent counts).

Both models learn from autocorrelation structure in lagged returns and order-flow
imbalance — the only signal that separates Val (mean-reverting) from Mom
(trending) episodes without seeing agent counts.

Models
------
  LSTMRegimeClassifier
    Single LSTM layer (hidden=32) + linear head → 2-class softmax.
    Input: rolling window of last SEQ_LEN=20 feature vectors.
    Handles temporal dependencies naturally; no manual feature engineering.

  MLPRegimeClassifier
    Two hidden layers (128 → 64) on a flattened SEQ_LEN×N_FEATURES window.
    Faster to train, interpretable weights, competitive with LSTM when
    the window is long enough to capture autocorrelation.

Features (6 per timestep, from the 7-dim daily-investor obs)
-------------------------------------------------------------
  [0] imbalance      = obs[1]       — bid vol / total vol; persistent in Mom
  [1] spread         = obs[2]/r_bar — bid-ask spread normalised
  [2] direction      = obs[3]/r_bar — mid − last transaction, normalised
  [3] ret_lag1       = obs[4]/r_bar — most-recent 60s return
  [4] ret_lag2       = obs[5]/r_bar — 2-step lag
  [5] ret_lag3       = obs[6]/r_bar — 3-step lag (3 lags already in obs)

Regime labels
-------------
  Val (num_momentum_agents=0)  → label 0
  Mom (num_momentum_agents=12) → label 1

Usage
-----
  # Train both models and save
  python regime_classifier.py --train-eps 60 --eval-eps 20 --save-dir models/

  # Quick test with fewer episodes
  python regime_classifier.py --train-eps 20 --eval-eps 10

Integration with PPO
--------------------
  from regime_classifier import TrainedRegimeBeliefWrapper, LSTMRegimeClassifier
  model = LSTMRegimeClassifier(); model.load_state_dict(torch.load("models/lstm.pt"))
  env = TrainedRegimeBeliefWrapper(base_env, model, model_type="lstm")
  # obs shape: (7+2, 1) — appends (p_val, p_mom)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gym
import abides_gym  # noqa: F401

from abides_gym.envs.markets_daily_investor_environment_v0 import (
    SubGymMarketsDailyInvestorEnv_v0,
)
from belief_tracker import KalmanBeliefTracker
from abides_core.utils import str_to_ns

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REGIME_VAL = {"num_momentum_agents": 0,  "num_value_agents": 102}
REGIME_MOM = {"num_momentum_agents": 12, "num_value_agents": 102}

LABEL_VAL = 0
LABEL_MOM = 1

R_BAR        = 100_000.0
SEQ_LEN      = 20          # timesteps of history fed to each model
N_FEATURES_BASE   = 6      # raw obs features
N_FEATURES_KALMAN = 8      # + belief_dev, belief_std
WARMUP_STEPS = 5           # skip env warmup where reward=0 and obs is uninformative

ENV_KW = dict(
    background_config         = "rmsc04",
    mkt_close                 = "16:00:00",
    timestep_duration         = "60s",
    starting_cash             = 1_000_000,
    order_fixed_size          = 10,
    state_history_length      = 4,
    market_data_buffer_length = 5,
    first_interval            = "00:05:00",
    reward_mode               = "dense",
    done_ratio                = 0.3,
    debug_mode                = False,
)

BG_KW = dict(
    r_bar              = int(R_BAR),
    kappa              = 1.67e-15,
    kappa_oracle       = 1.67e-16,
    lambda_a           = 5.7e-12,
    sigma_s            = 0,
    fund_vol           = 5e-5,
    megashock_lambda_a = 2.77778e-18,
    megashock_mean     = 1000,
    megashock_var      = 50_000,
    mm_window_size     = "adaptive",
    mm_pov             = 0.025,
    mm_num_ticks       = 10,
    mm_wake_up_freq    = "60S",
    mm_min_order_size  = 1,
    mm_skew_beta       = 0,
    mm_price_skew      = 4,
    mm_level_spacing   = 5,
    mm_spread_alpha    = 0.75,
    mm_backstop_quantity  = 0,
    mm_cancel_limit_delay = 50,
    num_noise_agents   = 1000,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    obs: np.ndarray,
    belief_dev: float = 0.0,
    belief_std: float = 0.0,
    use_kalman: bool  = False,
) -> np.ndarray:
    """
    6 or 8 normalised features from one daily-investor obs step.

    Base (6): imbalance, spread, direction, 3 lagged returns.
    Kalman (+2): belief_dev = (mu-mid)/r_bar, belief_std = sigma/r_bar.

    The Kalman features integrate 9-step price history into a smoother
    signal than the raw lags — helpful for distinguishing momentum
    (belief_dev drifts persistently in one direction) from value
    (belief_dev oscillates around zero).
    """
    o = obs.reshape(-1)
    feats = [
        float(o[1]),              # imbalance    [0, 1]
        float(o[2]) / R_BAR,     # spread       normalised
        float(o[3]) / R_BAR,     # direction    normalised
        float(o[4]) / R_BAR,     # ret_lag1     normalised
        float(o[5]) / R_BAR if len(o) > 5 else 0.0,  # ret_lag2
        float(o[6]) / R_BAR if len(o) > 6 else 0.0,  # ret_lag3
    ]
    if use_kalman:
        feats.append(float(belief_dev))   # already normalised by r_bar
        feats.append(float(belief_std))   # already normalised by r_bar
    return np.array(feats, dtype=np.float32)


def _make_kalman() -> KalmanBeliefTracker:
    """Default Kalman tracker matching the RMSC04 calibration."""
    dt_ns = int(str_to_ns("60s"))
    return KalmanBeliefTracker(
        r_bar    = int(R_BAR),
        kappa    = 1.67e-16,
        fund_vol = 5e-5,
        dt_ns    = dt_ns,
        obs_noise_frac = 0.005,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(regime_kw: dict) -> gym.Env:
    bg = dict(BG_KW)
    bg.update(regime_kw)
    kw = dict(ENV_KW)
    kw["background_config_extra_kvargs"] = bg
    return SubGymMarketsDailyInvestorEnv_v0(**kw)


def collect_episodes(
    n_episodes: int,
    regime: str,           # "val" or "mom"
    seed_offset: int = 0,
    use_kalman: bool = False,
    verbose: bool    = True,
) -> List[np.ndarray]:
    """
    Run hold-only episodes.  Returns list of feature arrays shape (T, N_FEATURES),
    one per episode (warmup steps removed).

    If use_kalman=True, runs a KalmanBeliefTracker in parallel and appends
    (belief_dev, belief_std) to each feature vector (8 features instead of 6).
    Mid-price is recovered from info["best_bid"] / info["best_ask"].
    """
    regime_kw = REGIME_VAL if regime == "val" else REGIME_MOM
    episodes: List[np.ndarray] = []

    # Kalman needs debug_mode=True for best_bid/best_ask in info
    env_kw = dict(ENV_KW)
    if use_kalman:
        env_kw["debug_mode"] = True

    env_kw["background_config_extra_kvargs"] = dict(BG_KW)
    env_kw["background_config_extra_kvargs"].update(regime_kw)
    env = SubGymMarketsDailyInvestorEnv_v0(**env_kw)

    kalman = _make_kalman() if use_kalman else None

    for i in range(n_episodes):
        seed = seed_offset + i
        env.seed(seed)
        obs = env.reset()
        env.previous_marked_to_market = env.starting_cash

        if kalman is not None:
            kalman.reset()
            mid = R_BAR
            mu, sigma = kalman.step(mid)

        feats: List[np.ndarray] = []
        done = False
        while not done:
            if kalman is not None:
                bdev = (mu - mid) / R_BAR
                bstd = sigma / R_BAR
                feats.append(extract_features(obs, bdev, bstd, use_kalman=True))
            else:
                feats.append(extract_features(obs))
            try:
                obs, _, done, info = env.step(1)
            except AssertionError:
                done = True
                break
            if kalman is not None:
                bid = info.get("best_bid")
                ask = info.get("best_ask")
                if bid is not None and ask is not None and ask > bid:
                    mid = (float(bid) + float(ask)) / 2.0
                mu, sigma = kalman.step(mid)

        feats = feats[WARMUP_STEPS:]
        if len(feats) >= SEQ_LEN:
            episodes.append(np.array(feats, dtype=np.float32))

        if verbose:
            print(f"  {regime} ep {i+1:>3}/{n_episodes}  steps={len(feats)}")

    env.close()
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    val_eps: List[np.ndarray],
    mom_eps: List[np.ndarray],
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length seq_len over every episode.

    Returns
    -------
    X : (N, seq_len, N_FEATURES)
    y : (N,) int — 0=val, 1=mom
    """
    X, y = [], []
    for label, episodes in [(LABEL_VAL, val_eps), (LABEL_MOM, mom_eps)]:
        for ep in episodes:
            T = len(ep)
            for t in range(T - seq_len + 1):
                X.append(ep[t : t + seq_len])
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class LSTMRegimeClassifier(nn.Module):
    """
    LSTM → linear head → 2-class logits.

    At inference the caller passes the rolling hidden state so the model
    has memory across the full episode, not just the last seq_len steps.
    During training we reset hidden state per window (simpler, generalises).
    """

    def __init__(
        self,
        input_size:  int = N_FEATURES_BASE,
        hidden_size: int = 32,
        num_layers:  int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, x: torch.Tensor, hx=None):
        # x: (batch, seq_len, n_features)
        out, hx = self.lstm(x, hx)
        logits = self.head(out[:, -1, :])   # last timestep
        return logits, hx

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        z = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (z, z.clone())

    def predict_proba(self, x: torch.Tensor, hx=None):
        logits, hx = self.forward(x, hx)
        return torch.softmax(logits, dim=-1), hx


class MLPRegimeClassifier(nn.Module):
    """
    Flatten (seq_len × n_features) → two hidden layers → 2-class logits.

    Simpler than LSTM: no sequential inductive bias, but the flat window
    contains enough autocorrelation signal when seq_len ≥ 10.
    """

    def __init__(
        self,
        seq_len:     int = SEQ_LEN,
        n_features:  int = N_FEATURES_BASE,
        hidden_size: int = 128,
    ):
        super().__init__()
        in_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )
        self.seq_len    = seq_len
        self.n_features = n_features

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, n_features)  or  (batch, seq_len*n_features)
        return self.net(x.reshape(x.shape[0], -1))

    def predict_proba(self, x: torch.Tensor):
        return torch.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_loop(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_epochs: int   = 40,
    lr: float       = 3e-4,
    batch_size: int = 256,
    model_type: str = "lstm",
) -> List[dict]:
    """Shared training loop for LSTM and MLP."""
    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    Xv = torch.from_numpy(X_va)
    yv = torch.from_numpy(y_va)

    history = []
    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in dl:
            optimizer.zero_grad()
            if model_type == "lstm":
                logits, _ = model(xb)
            else:
                logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if model_type == "lstm":
                val_logits, _ = model(Xv)
            else:
                val_logits = model(Xv)
            val_acc  = (val_logits.argmax(1) == yv).float().mean().item()
            val_loss = criterion(val_logits, yv).item()

        row = {
            "epoch":    epoch + 1,
            "tr_loss":  total_loss / n,
            "tr_acc":   correct / n,
            "val_loss": val_loss,
            "val_acc":  val_acc,
        }
        history.append(row)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}/{n_epochs}  "
                  f"tr_acc={row['tr_acc']:.3f}  val_acc={val_acc:.3f}  "
                  f"tr_loss={row['tr_loss']:.4f}  val_loss={val_loss:.4f}")

    return history


def train_lstm(
    X_tr, y_tr, X_va, y_va,
    hidden_size: int = 32,
    n_epochs: int    = 40,
    lr: float        = 3e-4,
    n_features: int  = N_FEATURES_BASE,
) -> LSTMRegimeClassifier:
    model = LSTMRegimeClassifier(input_size=n_features, hidden_size=hidden_size)
    print(f"  LSTM params: {sum(p.numel() for p in model.parameters()):,}")
    _train_loop(model, X_tr, y_tr, X_va, y_va,
                n_epochs=n_epochs, lr=lr, model_type="lstm")
    model.eval()
    return model


def train_mlp(
    X_tr, y_tr, X_va, y_va,
    hidden_size: int = 128,
    n_epochs: int    = 40,
    lr: float        = 3e-4,
    n_features: int  = N_FEATURES_BASE,
) -> MLPRegimeClassifier:
    model = MLPRegimeClassifier(n_features=n_features, hidden_size=hidden_size)
    print(f"  MLP params:  {sum(p.numel() for p in model.parameters()):,}")
    _train_loop(model, X_tr, y_tr, X_va, y_va,
                n_epochs=n_epochs, lr=lr, model_type="mlp")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    val_eps: List[np.ndarray],
    mom_eps: List[np.ndarray],
    model_type: str = "lstm",
    seq_len: int    = SEQ_LEN,
) -> dict:
    """
    Episode-level accuracy: for each eval episode, take the mean predicted
    p_val and p_mom across all windows, then classify by argmax.
    Also reports mean per-step probabilities for each regime.
    """
    model.eval()
    results = {"val": [], "mom": []}

    for label, eps, key in [(LABEL_VAL, val_eps, "val"), (LABEL_MOM, mom_eps, "mom")]:
        for ep in eps:
            T = len(ep)
            p_vals_ep, p_moms_ep = [], []
            for t in range(T - seq_len + 1):
                window = torch.from_numpy(ep[t : t + seq_len]).unsqueeze(0)
                with torch.no_grad():
                    if model_type == "lstm":
                        proba, _ = model.predict_proba(window)
                    else:
                        proba = model.predict_proba(window)
                p_vals_ep.append(proba[0, 0].item())
                p_moms_ep.append(proba[0, 1].item())

            ep_p_val = float(np.mean(p_vals_ep))
            ep_p_mom = float(np.mean(p_moms_ep))
            predicted = "val" if ep_p_val > ep_p_mom else "mom"
            results[key].append({
                "p_val": ep_p_val,
                "p_mom": ep_p_mom,
                "correct": predicted == key,
            })

    val_acc = np.mean([r["correct"] for r in results["val"]])
    mom_acc = np.mean([r["correct"] for r in results["mom"]])
    overall = np.mean([r["correct"] for rs in results.values() for r in rs])

    mean_p_val_in_val = np.mean([r["p_val"] for r in results["val"]])
    mean_p_mom_in_mom = np.mean([r["p_mom"] for r in results["mom"]])

    return {
        "val_acc":         val_acc,
        "mom_acc":         mom_acc,
        "overall_acc":     overall,
        "mean_p_val|val":  mean_p_val_in_val,   # want this near 1.0
        "mean_p_mom|mom":  mean_p_mom_in_mom,   # want this near 1.0
    }


def print_eval(name: str, metrics: dict) -> None:
    print(f"\n  {name} evaluation:")
    print(f"    Val  episodes:  acc={metrics['val_acc']:.1%}  "
          f"mean p_val={metrics['mean_p_val|val']:.3f}")
    print(f"    Mom  episodes:  acc={metrics['mom_acc']:.1%}  "
          f"mean p_mom={metrics['mean_p_mom|mom']:.3f}")
    print(f"    Overall acc:    {metrics['overall_acc']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Gym wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TrainedRegimeBeliefWrapper(gym.Wrapper):
    """
    Appends (p_val_t, p_mom_t) from a trained LSTM or MLP to every obs.

    Same 2-feature obs contract as OracleRegimeWrapper — drop-in for PPO.
    At each step, the wrapper maintains a rolling buffer of the last SEQ_LEN
    feature vectors and queries the model.

    Parameters
    ----------
    env : gym.Env
    model : LSTMRegimeClassifier or MLPRegimeClassifier
    model_type : str — "lstm" or "mlp"
    seq_len : int — window length (must match training)
    use_kalman : bool — if True, also runs a Kalman tracker per step and
        appends (belief_dev, belief_std) to each feature vector.
        Must match the use_kalman setting used during training.
    """

    def __init__(
        self,
        env: gym.Env,
        model: nn.Module,
        model_type:  str  = "lstm",
        seq_len:     int  = SEQ_LEN,
        use_kalman:  bool = False,
    ):
        super().__init__(env)
        self.model      = model
        self.model.eval()
        self.model_type = model_type
        self.seq_len    = seq_len
        self.use_kalman = use_kalman
        self._buf: List[np.ndarray] = []
        self._hx = None

        self._kalman: Optional[KalmanBeliefTracker] = (
            _make_kalman() if use_kalman else None
        )
        self._mid  = R_BAR
        self._bdev = 0.0
        self._bstd = 0.0

        n_feat = N_FEATURES_KALMAN if use_kalman else N_FEATURES_BASE

        old_low  = env.observation_space.low
        old_high = env.observation_space.high
        extra_low  = np.full((2, 1), 0.0, dtype=np.float32)
        extra_high = np.full((2, 1), 1.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low  = np.vstack([old_low,  extra_low]),
            high = np.vstack([old_high, extra_high]),
            dtype=np.float32,
        )
        self._n_feat = n_feat

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        if self._kalman is not None:
            self._kalman.reset()
            self._mid  = R_BAR
            mu, sigma  = self._kalman.step(self._mid)
            self._bdev = (mu - self._mid) / R_BAR
            self._bstd = sigma / R_BAR
        self._buf = [extract_features(obs, self._bdev, self._bstd, self.use_kalman)]
        self._hx  = None
        p_val, p_mom = self._predict()
        return self._augment(obs, p_val, p_mom)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self._kalman is not None:
            bid = info.get("best_bid")
            ask = info.get("best_ask")
            if bid is not None and ask is not None and ask > bid:
                self._mid = (float(bid) + float(ask)) / 2.0
            mu, sigma  = self._kalman.step(self._mid)
            self._bdev = (mu - self._mid) / R_BAR
            self._bstd = sigma / R_BAR
        self._buf.append(
            extract_features(obs, self._bdev, self._bstd, self.use_kalman)
        )
        if len(self._buf) > self.seq_len:
            self._buf.pop(0)
        p_val, p_mom = self._predict()
        return self._augment(obs, p_val, p_mom), reward, done, info

    def _predict(self) -> Tuple[float, float]:
        buf = list(self._buf)
        if len(buf) < self.seq_len:
            pad = [np.zeros(self._n_feat, dtype=np.float32)] * (self.seq_len - len(buf))
            buf = pad + buf
        x = torch.from_numpy(np.array(buf, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            if self.model_type == "lstm":
                proba, self._hx = self.model.predict_proba(x, self._hx)
            else:
                proba = self.model.predict_proba(x)
        return float(proba[0, 0].item()), float(proba[0, 1].item())

    def _augment(self, obs: np.ndarray, p_val: float, p_mom: float) -> np.ndarray:
        extra = np.array([[np.float32(p_val)], [np.float32(p_mom)]], dtype=np.float32)
        return np.vstack([obs, extra])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(path: str):
    """Load a trained regime classifier from a checkpoint file.

    Returns (model, model_type, use_kalman, seq_len).
    Compatible with checkpoints produced by this script's --save-dir logic.
    """
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    mtype      = ckpt["model_type"]
    use_kalman = ckpt["use_kalman"]
    seq_len    = ckpt["seq_len"]
    n_features = ckpt["n_features"]
    hidden     = ckpt["hidden_size"]

    if mtype == "lstm":
        model = LSTMRegimeClassifier(input_size=n_features, hidden_size=hidden)
    else:
        model = MLPRegimeClassifier(seq_len=seq_len, n_features=n_features, hidden_size=hidden)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mtype, use_kalman, seq_len


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-eps",  type=int, default=60,
                   help="Episodes per regime for training (default 60)")
    p.add_argument("--eval-eps",   type=int, default=20,
                   help="Episodes per regime for evaluation (default 20)")
    p.add_argument("--seq-len",    type=int, default=SEQ_LEN)
    p.add_argument("--n-epochs",   type=int, default=40)
    p.add_argument("--hidden",     type=int, default=32,
                   help="LSTM hidden size (MLP uses 4x this)")
    p.add_argument("--save-dir",   type=str, default="models",
                   help="Directory to save trained models")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--no-kalman",  action="store_true",
                   help="Skip Kalman-augmented variants")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("Regime classifier training")
    print("=" * 64)
    print(f"  train episodes : {args.train_eps} per regime  "
          f"({2 * args.train_eps} total)")
    print(f"  eval  episodes : {args.eval_eps} per regime")
    print(f"  seq_len        : {args.seq_len}")
    print(f"  n_epochs       : {args.n_epochs}")
    print()

    TRAIN_OFFSET = 200
    EVAL_OFFSET  = 300

    all_results = {}

    for use_kalman in ([False, True] if not args.no_kalman else [False]):
        suffix     = "+Kalman" if use_kalman else "raw"
        n_feat     = N_FEATURES_KALMAN if use_kalman else N_FEATURES_BASE

        print(f"\n{'='*64}")
        print(f"Feature set: {suffix}  ({n_feat} features/step)")
        print(f"{'='*64}")

        print("Collecting training episodes ...")
        tr_val = collect_episodes(args.train_eps, "val",
                                  seed_offset=TRAIN_OFFSET, use_kalman=use_kalman)
        tr_mom = collect_episodes(args.train_eps, "mom",
                                  seed_offset=TRAIN_OFFSET, use_kalman=use_kalman)

        print("\nCollecting eval episodes ...")
        ev_val = collect_episodes(args.eval_eps, "val",
                                  seed_offset=EVAL_OFFSET, use_kalman=use_kalman)
        ev_mom = collect_episodes(args.eval_eps, "mom",
                                  seed_offset=EVAL_OFFSET, use_kalman=use_kalman)

        X_tr, y_tr = build_dataset(tr_val, tr_mom, seq_len=args.seq_len)
        X_va, y_va = build_dataset(ev_val, ev_mom, seq_len=args.seq_len)
        val_frac   = (y_tr == LABEL_VAL).mean()
        print(f"\nDataset: train={len(X_tr):,}  eval={len(X_va):,}  "
              f"shape={X_tr.shape}  balance val={val_frac:.0%}/mom={1-val_frac:.0%}")

        # LSTM
        tag = f"LSTM ({suffix})"
        print(f"\n── {tag} ──")
        lstm_m = train_lstm(X_tr, y_tr, X_va, y_va,
                            hidden_size=args.hidden,
                            n_epochs=args.n_epochs, n_features=n_feat)
        lstm_metrics = evaluate(lstm_m, ev_val, ev_mom,
                                model_type="lstm", seq_len=args.seq_len)
        print_eval(tag, lstm_metrics)
        all_results[tag] = (lstm_metrics, lstm_m, "lstm", use_kalman)

        # MLP
        tag = f"MLP  ({suffix})"
        print(f"\n── {tag} ──")
        mlp_m = train_mlp(X_tr, y_tr, X_va, y_va,
                          hidden_size=args.hidden * 4,
                          n_epochs=args.n_epochs, n_features=n_feat)
        mlp_metrics = evaluate(mlp_m, ev_val, ev_mom,
                               model_type="mlp", seq_len=args.seq_len)
        print_eval(tag, mlp_metrics)
        all_results[tag] = (mlp_metrics, mlp_m, "mlp", use_kalman)

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  {'Model':<22}  {'Overall acc':>12}  "
          f"{'p_val|val':>10}  {'p_mom|mom':>10}")
    print("=" * 72)
    for name, (m, _, _, _) in all_results.items():
        print(f"  {name:<22}  {m['overall_acc']:>12.1%}  "
              f"{m['mean_p_val|val']:>10.3f}  {m['mean_p_mom|mom']:>10.3f}")
    print("=" * 72)

    # ── save ──────────────────────────────────────────────────────────────────
    if not args.no_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, (_, model, mtype, use_k) in all_results.items():
            fname = f"{'lstm' if mtype=='lstm' else 'mlp'}_{'kalman' if use_k else 'raw'}_regime.pt"
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_type": mtype,
                "use_kalman": use_k,
                "seq_len":    args.seq_len,
                "n_features": N_FEATURES_KALMAN if use_k else N_FEATURES_BASE,
                "hidden_size": args.hidden if mtype == "lstm" else args.hidden * 4,
            }
            torch.save(checkpoint, save_dir / fname)
        print(f"\nModels saved → {save_dir}/")


if __name__ == "__main__":
    main()
