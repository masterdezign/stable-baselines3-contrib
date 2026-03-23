"""
Train RecurrentSAC on LunarLanderContinuousNoVel-v3 for 5 M steps.

LunarLander-v3 (continuous) observations:
    [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

NoVel wrapper removes velocity components (vx, vy, angular_vel), leaving
    [x, y, angle, left_leg_contact, right_leg_contact]  — 5 observations.
The agent must integrate position over time via its LSTM to infer velocity.

Logging:
    - Training rollout stats (ep_rew_mean, ep_len_mean) from SB3 logger
    - Losses: actor, critic, ent_coef, ent_coef_loss
    - Periodic deterministic evaluation on a separate eval env (raw reward)
    All written to <run_dir>/log.csv

Visualization:
    Reads log.csv and saves <run_dir>/plots.png

Usage:
    # Train (creates results/run_<timestamp>/)
    PYTHONPATH=. python examples/train_lunarlander.py

    # Train with custom options
    PYTHONPATH=. python examples/train_lunarlander.py --n-envs 8 --seed 1

    # Plot an existing log
    PYTHONPATH=. python examples/train_lunarlander.py --mode plot --run-dir results/run_20240101_120000
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentSAC

# Original obs: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
# Keep:          x=0  y=1             angle=4       left=6    right=7
_KEEP_IDX = np.array([0, 1, 4, 6, 7])


class LunarLanderNoVel(gym.ObservationWrapper):
    """Remove vx, vy, and angular_vel from LunarLanderContinuous-v3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low[_KEEP_IDX]
        high = env.observation_space.high[_KEEP_IDX]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[_KEEP_IDX].astype(np.float32)


def _make_env(**kwargs) -> gym.Env:
    return LunarLanderNoVel(gym.make("LunarLanderContinuous-v3", **kwargs))


def make_venv(n_envs: int, seed: int = 0) -> VecNormalize:
    env = make_vec_env(_make_env, n_envs=n_envs, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_eval_venv(seed: int = 0) -> VecNormalize:
    env = make_vec_env(_make_env, n_envs=1, seed=seed)
    # norm_reward=False so we get raw (human-interpretable) episode returns
    return VecNormalize(env, norm_obs=True, norm_reward=False, training=False)


_CSV_FIELDS = [
    "timestep",
    "wall_time_s",
    "rollout_ep_rew_mean",
    "rollout_ep_len_mean",
    "eval_mean_reward",
    "eval_std_reward",
    "actor_loss",
    "critic_loss",
    "ent_coef",
    "ent_coef_loss",
    "n_updates",
]

# Mapping from SB3 logger key → CSV field name
_LOGGER_MAP = {
    "rollout/ep_rew_mean": "rollout_ep_rew_mean",
    "rollout/ep_len_mean": "rollout_ep_len_mean",
    "train/actor_loss": "actor_loss",
    "train/critic_loss": "critic_loss",
    "train/ent_coef": "ent_coef",
    "train/ent_coef_loss": "ent_coef_loss",
    "train/n_updates": "n_updates",
}


class LoggingCallback(BaseCallback):
    """
    Periodically logs training metrics and deterministic eval reward to CSV.

    The callback caches the latest value seen for each SB3 logger key
    (losses are only emitted after gradient updates, so caching ensures they
    are available at arbitrary timestep intervals).
    """

    def __init__(
        self,
        eval_env: VecNormalize,
        csv_path: str | Path,
        log_freq: int = 25_000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = Path(csv_path)
        self.log_freq = log_freq
        self.n_eval_episodes = n_eval_episodes

        self._cached: dict[str, float] = {}
        self._last_log_step = -log_freq  # log immediately at step 0
        self._start_time: float = 0.0
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        if self.verbose:
            print(f"[LoggingCallback] logging to {self.csv_path}", flush=True)

    def _on_step(self) -> bool:
        # Cache latest values from the SB3 logger (cleared on each dump())
        for sb3_key, csv_key in _LOGGER_MAP.items():
            if sb3_key in self.model.logger.name_to_value:
                self._cached[csv_key] = self.model.logger.name_to_value[sb3_key]

        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._last_log_step = self.num_timesteps
            self._write_row()

        return True

    def _on_training_end(self) -> None:
        # Final row at the very end of training
        if self.num_timesteps != self._last_log_step:
            self._write_row()
        if self._file is not None:
            self._file.close()
        if self.verbose:
            print("[LoggingCallback] training finished, log closed.", flush=True)

    def _sync_eval_env(self) -> None:
        """Copy normalisation statistics from the training env to eval_env."""
        train_env: VecNormalize = self.training_env  # type: ignore[assignment]
        self.eval_env.obs_rms = train_env.obs_rms
        self.eval_env.ret_rms = train_env.ret_rms

    def _write_row(self) -> None:
        self._sync_eval_env()

        eval_mean, eval_std = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )

        row: dict[str, object] = {
            "timestep": self.num_timesteps,
            "wall_time_s": round(time.time() - self._start_time, 1),
            "eval_mean_reward": round(float(eval_mean), 3),
            "eval_std_reward": round(float(eval_std), 3),
        }
        row.update({k: self._cached.get(k) for k in _CSV_FIELDS if k not in row})

        self._writer.writerow(row)
        self._file.flush()

        if self.verbose:
            rollout_str = ""
            if "rollout_ep_rew_mean" in self._cached:
                rollout_str = f"  rollout={self._cached['rollout_ep_rew_mean']:.1f}"
            actor_str = ""
            if "actor_loss" in self._cached:
                actor_str = (
                    f"  actor={self._cached['actor_loss']:.4f}  critic={self._cached.get('critic_loss', float('nan')):.4f}"
                )
            print(
                f"  t={self.num_timesteps:>9,}  eval={eval_mean:.1f}±{eval_std:.1f}" f"{rollout_str}{actor_str}",
                flush=True,
            )


def train(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "log.csv"

    print(f"Run dir : {run_dir}", flush=True)
    print(f"Timesteps: {args.total_timesteps:,}", flush=True)
    print(f"n_envs   : {args.n_envs}", flush=True)
    print(f"seed     : {args.seed}", flush=True)

    train_env = make_venv(n_envs=args.n_envs, seed=args.seed)
    eval_env = make_eval_venv(seed=args.seed + 1000)

    # Best hyperparameters from Optuna (tune_rsac_lunarlander2.py, 1M budget/trial)
    model = RecurrentSAC(
        "MlpLstmPolicy",
        train_env,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        ent_coef="auto",
        train_freq=4,
        gradient_steps=8,
        segment_len=32,
        overlap=10,
        burn_in=4,
        shared_state=True,
        buffer_size=100_000,
        policy_kwargs={
            "net_arch": [128, 128, 128],
            "lstm_hidden_size": 64,
            "n_lstm_layers": 2,
        },
        verbose=0,
        seed=args.seed,
    )

    callback = LoggingCallback(
        eval_env=eval_env,
        csv_path=csv_path,
        log_freq=args.log_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/3600:.2f} h  ({elapsed:.0f} s)", flush=True)

    model.save(run_dir / "model")
    train_env.save(run_dir / "vec_normalize.pkl")

    train_env.close()
    eval_env.close()

    return csv_path


_SMOOTH_WINDOW = 15  # rows (not timesteps) for rolling mean


def _smooth(s: pd.Series, window: int = _SMOOTH_WINDOW) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def plot(run_dir: Path) -> None:
    csv_path = run_dir / "log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    t = df["timestep"] / 1e6  # x-axis in millions of steps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"RecurrentSAC — LunarLanderContinuousNoVel-v3\n{run_dir.name}",
        fontsize=13,
    )

    # ------------------------------------------------------------------
    # 1. Eval reward
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ax.fill_between(
        t,
        df["eval_mean_reward"] - df["eval_std_reward"],
        df["eval_mean_reward"] + df["eval_std_reward"],
        alpha=0.2,
        label="_nolegend_",
    )
    ax.plot(t, df["eval_mean_reward"], lw=1.5, label="eval (det.)")
    if df["rollout_ep_rew_mean"].notna().any():
        ax.plot(t, _smooth(df["rollout_ep_rew_mean"]), lw=1, alpha=0.7, label="train rollout (smooth)")
    ax.axhline(200, color="green", ls="--", lw=1, label="solved (200)")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 2. Actor loss
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    if df["actor_loss"].notna().any():
        ax.plot(t, _smooth(df["actor_loss"]), lw=1.5)
    ax.set_title("Actor Loss (smoothed)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 3. Critic loss
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    if df["critic_loss"].notna().any():
        ax.plot(t, _smooth(df["critic_loss"]), lw=1.5, color="tab:orange")
    ax.set_title("Critic Loss (smoothed)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 4. Entropy coefficient
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if df["ent_coef"].notna().any():
        ax.plot(t, df["ent_coef"], lw=1.5, color="tab:green")
    ax.set_title("Entropy Coefficient (α)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("α")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 5. Entropy coefficient loss
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if df["ent_coef_loss"].notna().any():
        ax.plot(t, _smooth(df["ent_coef_loss"]), lw=1.5, color="tab:red")
    ax.set_title("Entropy Coefficient Loss (smoothed)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 6. Episode length
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    if df["rollout_ep_len_mean"].notna().any():
        ax.plot(t, _smooth(df["rollout_ep_len_mean"]), lw=1.5, color="tab:purple")
    ax.set_title("Episode Length — train rollout (smoothed)")
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = run_dir / "plots.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot → {out}", flush=True)
    plt.show()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    eval_col = df["eval_mean_reward"].dropna()
    if not eval_col.empty:
        best_idx = eval_col.idxmax()
        best_rew = eval_col.iloc[best_idx]
        best_t = df["timestep"].iloc[best_idx]
        final_rew = eval_col.iloc[-1]
        print(f"\nSummary:")
        print(f"  Best eval reward : {best_rew:.1f}  at {best_t:,} steps")
        print(f"  Final eval reward: {final_rew:.1f}")
        print(f"  Solved (≥200)    : {'yes' if best_rew >= 200 else 'no'}")


def _default_run_dir() -> str:
    return f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RecurrentSAC on LunarLanderContinuousNoVel-v3")
    parser.add_argument("--mode", choices=["train", "plot", "both"], default="both")
    parser.add_argument("--run-dir", type=str, default=_default_run_dir(), help="Output directory")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-freq",
        type=int,
        default=12_500,
        help="Log / eval every N timesteps (default: 12.5k → 400 points for 5M run)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = Path(args.run_dir)

    if args.mode in ("train", "both"):
        train(args)

    if args.mode in ("plot", "both"):
        plot(run_dir)
