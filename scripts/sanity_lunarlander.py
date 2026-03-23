"""
Sanity check: RecurrentSAC on standard LunarLanderContinuous-v3 (full obs).

Full observation: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg] — 8 obs.
No velocity hidden — this should be easier than the NoVel variant.
Solved threshold: mean reward >= 200.

Usage:
    PYTHONPATH=. python scripts/sanity_lunarlander.py
    PYTHONPATH=. python scripts/sanity_lunarlander.py --timesteps 200000 --seed 1
"""

import argparse
import time

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentSAC

SOLVED_THRESHOLD = 200.0
N_EVAL_EPISODES = 20


def make_env(n_envs: int, seed: int) -> VecNormalize:
    env = make_vec_env("LunarLanderContinuous-v3", n_envs=n_envs, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def make_eval_env(seed: int) -> VecNormalize:
    env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=False, training=False)


def run(args: argparse.Namespace) -> None:
    print(f"RecurrentSAC sanity check — LunarLanderContinuous-v3 (full obs)")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  seed      : {args.seed}")
    print(flush=True)

    train_env = make_env(n_envs=args.n_envs, seed=args.seed)
    eval_env = make_eval_env(seed=args.seed + 1000)

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
        verbose=1,
        seed=args.seed,
    )

    checkpoints = np.linspace(0, args.timesteps, args.n_checkpoints + 1, dtype=int)[1:]
    prev_steps = 0
    t0 = time.time()

    for checkpoint in checkpoints:
        model.learn(total_timesteps=int(checkpoint) - prev_steps, reset_num_timesteps=False)
        prev_steps = int(checkpoint)

        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )

        elapsed = time.time() - t0
        solved = "  *** SOLVED ***" if mean_reward >= SOLVED_THRESHOLD else ""
        print(
            f"  [{checkpoint:>7,} steps | {elapsed:5.0f}s]  "
            f"eval = {mean_reward:7.1f} ± {std_reward:.1f}{solved}",
            flush=True,
        )

    train_env.close()
    eval_env.close()

    result = "PASSED" if mean_reward >= SOLVED_THRESHOLD else "FAILED"
    print(f"\nSanity check {result} "
          f"(final reward {mean_reward:.1f}, threshold {SOLVED_THRESHOLD})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-checkpoints", type=int, default=10,
                        help="How many intermediate eval points to print")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args)
