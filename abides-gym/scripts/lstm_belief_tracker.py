"""
lstm_belief_tracker.py — LSTM-based belief tracker for ABIDES daily investor
=============================================================================

Learns to estimate short-term price drift from market feature history,
without knowing the OU parameters (kappa, fund_vol).

Design
------
  Input  : rolling window of T=20 timesteps, each with 4 features:
             [return/r_bar, imbalance, spread/r_bar, direction/r_bar]
  Output : predicted N-step future return (normalized by r_bar)
             > 0 → expect price to rise → BUY signal
             < 0 → expect price to fall → SELL signal
  Training: self-supervised on hold-only rollouts
             label_t = mean(mid_{t+1..t+N}) - mid_t, divided by r_bar

Contrast with Kalman
--------------------
  Kalman knows kappa_oracle and fund_vol — it implements the theoretically
  optimal filter for the OU model with those exact params.
  LSTM knows nothing about the model — it learns from price history alone.
  Advantage of LSTM: generalises across unknown / varying parameters.
  Disadvantage    : needs training data; black-box.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

import gym
import abides_gym  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(obs: np.ndarray, r_bar: float) -> np.ndarray:
    """
    4-dim feature vector from one observation step.
    All values normalised so they're O(1) regardless of r_bar.
    """
    ret       = float(obs[4, 0]) / r_bar if obs.shape[0] > 4 else 0.0
    imbalance = float(obs[1, 0])                # already [0, 1]
    spread    = float(obs[2, 0]) / r_bar
    direction = float(obs[3, 0]) / r_bar
    return np.array([ret, imbalance, spread, direction], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBeliefNet(nn.Module):
    """
    Single-layer LSTM followed by a linear head.

    forward(x) -> (prediction, hidden_state)
      x: (batch, seq_len, 4)
      prediction: (batch,) — predicted future return / r_bar
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hx=None):
        out, hx = self.lstm(x, hx)          # out: (batch, seq, hidden)
        pred = self.head(out[:, -1, :])     # last timestep only
        return pred.squeeze(-1), hx

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollouts(
    n_episodes: int = 30,
    r_bar: int = 100_000,
    timestep: str = "60s",
    bg_kwargs: dict = None,
    seed_offset: int = 200,
    verbose: bool = True,
) -> list:
    """
    Run hold-only episodes. Returns list of dicts with 'features' and 'mid_prices'.
    Uses seeds [seed_offset, seed_offset + n_episodes) to avoid overlap with test seeds.
    """
    if bg_kwargs is None:
        bg_kwargs = {}

    env = gym.make(
        "markets-daily_investor-v0",
        background_config="rmsc04",
        debug_mode=True,
        timestep_duration=timestep,
        background_config_extra_kvargs=bg_kwargs,
    )

    episodes = []
    for i in range(n_episodes):
        seed = seed_offset + i
        env.seed(seed)
        obs = env.reset()
        env.previous_marked_to_market = env.starting_cash

        feats, mids = [], []
        done = False
        while not done:
            feats.append(extract_features(obs, r_bar))
            try:
                obs, _, done, info = env.step(1)   # always hold
            except AssertionError:
                done = True
                break
            bid = info.get("best_bid", r_bar)
            ask = info.get("best_ask", r_bar)
            mids.append((bid + ask) / 2.0 if ask > bid else float(r_bar))

        if verbose:
            print(f"  rollout {i+1:>2}/{n_episodes}  len={len(feats)}")

        if len(feats) > 1 and len(mids) > 0:
            episodes.append({
                "features":   np.array(feats, dtype=np.float32),
                "mid_prices": np.array(mids,  dtype=np.float32),
            })

    env.close()
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    episodes: list,
    seq_len: int = 20,
    lookahead: int = 3,
    r_bar: float = 100_000,
):
    """
    X: (N, seq_len, 4) — feature windows
    y: (N,)            — mean future return over lookahead steps, / r_bar
    """
    X, y = [], []
    for ep in episodes:
        feats = ep["features"]    # (T, 4)
        mids  = ep["mid_prices"]  # (T,) — one entry per step() call
        T     = min(len(feats), len(mids))
        if T < lookahead + 1:
            continue

        for t in range(T - lookahead):
            # Feature window: last seq_len steps up to t (inclusive)
            start  = max(0, t + 1 - seq_len)
            window = feats[start : t + 1]
            if len(window) < seq_len:
                pad    = np.zeros((seq_len - len(window), 4), dtype=np.float32)
                window = np.vstack([pad, window])
            X.append(window)

            # Label: average return over next 'lookahead' mid-price steps
            future = mids[t : t + lookahead + 1]
            label  = float(np.mean(np.diff(future))) / r_bar
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    episodes: list,
    seq_len: int    = 20,
    lookahead: int  = 3,
    r_bar: float    = 100_000,
    hidden_size: int = 32,
    n_epochs: int   = 30,
    lr: float       = 1e-3,
    batch_size: int = 256,
) -> "LSTMBeliefNet":
    """Train and return an LSTMBeliefNet."""
    X, y = build_dataset(episodes, seq_len, lookahead, r_bar)
    print(f"  dataset: {len(X)} samples  (seq_len={seq_len}, lookahead={lookahead})")

    model     = LSTMBeliefNet(hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>3}/{n_epochs}  mse={total/len(X):.8f}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Gym wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBeliefWrapper(gym.Wrapper):
    """
    Appends LSTM belief_dev to the observation, same shape contract as
    BeliefAugmentedWrapper — new obs is (original_features + 2, 1).

      obs[-2] = lstm_belief_dev  (predicted future return / r_bar)
      obs[-1] = 0.0              (LSTM gives no uncertainty estimate)

    The wrapper maintains a rolling buffer of the last seq_len feature
    vectors and queries the model at every step.
    """

    def __init__(
        self,
        env: gym.Env,
        model: LSTMBeliefNet,
        seq_len: int = 20,
        r_bar: int = 100_000,
    ):
        super().__init__(env)
        self.model   = model
        self.model.eval()
        self.seq_len = seq_len
        self.r_bar   = float(r_bar)
        self._buf: list = []

        old_low  = env.observation_space.low
        old_high = env.observation_space.high
        self.observation_space = gym.spaces.Box(
            low  = np.vstack([old_low,  np.full((2, 1), -20.0, dtype=np.float32)]),
            high = np.vstack([old_high, np.full((2, 1),  20.0, dtype=np.float32)]),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        self.env.previous_marked_to_market = self.env.starting_cash
        self._buf = [extract_features(obs, self.r_bar)]
        return self._augment(obs)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self._buf.append(extract_features(obs, self.r_bar))
        if len(self._buf) > self.seq_len:
            self._buf.pop(0)
        return self._augment(obs), reward, done, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        buf = list(self._buf)
        if len(buf) < self.seq_len:
            pad = [np.zeros(4, dtype=np.float32)] * (self.seq_len - len(buf))
            buf = pad + buf
        x = torch.from_numpy(np.array(buf, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred, _ = self.model(x)
        belief_dev = float(pred.item())
        extra = np.array([[np.float32(belief_dev)], [0.0]], dtype=np.float32)
        return np.vstack([obs, extra])
