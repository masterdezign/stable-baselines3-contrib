"""
Train RecurrentSAC and RecurrentPPO on PendulumNoVel-v1.

    python examples/pendulum_no_vel.py
    python examples/pendulum_no_vel.py --algo rsac
    python examples/pendulum_no_vel.py --algo ppo

PendulumNoVel removes the angular velocity observation, turning
Pendulum-v1 into a POMDP. The agent must integrate angular position
over time with its LSTM to infer velocity.

Benchmark results (50k steps, 2 seeds, n_envs=4):
    RecurrentSAC  mean: -745  (seeds: -724, -766)
    RecurrentPPO  mean: -803  (seeds: -798, -807)
    Random policy mean: ~-1400
    Solved:             ~-150
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentPPO, RecurrentSAC


# ---------------------------------------------------------------------------
# PendulumNoVel wrapper
# ---------------------------------------------------------------------------

class PendulumNoVel(gym.ObservationWrapper):
    """Remove the angular-velocity component from Pendulum-v1 observations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Original obs: [cos θ, sin θ, ω]  →  keep first two only
        low = env.observation_space.low[:2]
        high = env.observation_space.high[:2]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[:2]


def make_pendulum_no_vel(**kwargs):
    env = gym.make("Pendulum-v1", **kwargs)
    return PendulumNoVel(env)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_rsac(n_envs: int = 4, total_timesteps: int = 100_000, seed: int = 42) -> float:
    """Train RecurrentSAC and return mean evaluation reward."""
    vec_env = make_vec_env(make_pendulum_no_vel, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = RecurrentSAC(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        ent_coef="auto",
        # 4:1 gradient-to-env-step ratio drives sample efficiency on a short budget
        train_freq=4,
        gradient_steps=16,
        # Sequence chunks: 32 steps with 8-step overlap and 8-step burn-in
        segment_len=32,
        overlap=8,
        burn_in=8,
        # shared_state=True: critic reuses actor LSTM output; single LSTM to learn
        shared_state=True,
        buffer_size=100_000,
        policy_kwargs={"net_arch": [64, 64, 64], "lstm_hidden_size": 64, "n_lstm_layers": 1},
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Evaluate on unnormalised env
    eval_env = make_vec_env(make_pendulum_no_vel, n_envs=1, seed=seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Copy running statistics from training env
    eval_env.obs_rms = vec_env.obs_rms

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\nRecurrentSAC  — mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    return mean_reward


def train_ppo(n_envs: int = 4, total_timesteps: int = 100_000, seed: int = 42) -> float:
    """Train RecurrentPPO and return mean evaluation reward."""
    vec_env = make_vec_env(make_pendulum_no_vel, n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=1e-3,
        gamma=0.99,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"lstm_hidden_size": 64, "net_arch": [64, 64, 64]},
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    eval_env = make_vec_env(make_pendulum_no_vel, n_envs=1, seed=seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = vec_env.obs_rms

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\nRecurrentPPO  — mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    return mean_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["rsac", "ppo", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps (default 100k)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    if args.algo in ("rsac", "both"):
        ts = args.timesteps or 100_000
        train_rsac(n_envs=args.n_envs, total_timesteps=ts, seed=args.seed)

    if args.algo in ("ppo", "both"):
        ts = args.timesteps or 100_000
        train_ppo(n_envs=args.n_envs, total_timesteps=ts, seed=args.seed)


if __name__ == "__main__":
    main()
