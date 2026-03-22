"""
Optuna hyperparameter search for RecurrentSAC on LunarLanderContinuousNoVel-v3.

LunarLander-v3 (continuous) observations:
    [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]

NoVel wrapper removes the velocity components (vx, vy, angular_vel),
leaving [x, y, angle, left_leg_contact, right_leg_contact] — 5 observations.
The agent must integrate position over time with its LSTM to infer velocity.

Solved threshold: mean episode reward ≥ 200.

Usage:
    PYTHONPATH=. python scripts/tune_rsac_lunarlander.py
    PYTHONPATH=. python scripts/tune_rsac_lunarlander.py --n-trials 40 --n-timesteps 300000
    PYTHONPATH=. python scripts/tune_rsac_lunarlander.py --storage sqlite:///lunarlander.db --n-trials 50
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentSAC

# Indices kept after removing velocity observations
# Original: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
_KEEP_IDX = np.array([0, 1, 4, 6, 7])  # x, y, angle, left_leg, right_leg

N_EVAL_EPISODES = 10
SOLVED_THRESHOLD = 200.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LunarLanderNoVel(gym.ObservationWrapper):
    """Remove vx, vy, and angular_vel from LunarLanderContinuous-v3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = env.observation_space.low[_KEEP_IDX]
        high = env.observation_space.high[_KEEP_IDX]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs[_KEEP_IDX]


def make_lunar_no_vel(**kwargs) -> gym.Env:
    return LunarLanderNoVel(gym.make("LunarLanderContinuous-v3", **kwargs))


def make_env(n_envs: int, seed: int = 0) -> VecNormalize:
    env = make_vec_env(make_lunar_no_vel, n_envs=n_envs, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=True)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------


def sample_params(trial: optuna.Trial) -> dict:
    """Sample RecurrentSAC hyperparameters for LunarLander."""
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99])
    tau = trial.suggest_categorical("tau", [0.005, 0.01])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.1])
    n_envs = trial.suggest_categorical("n_envs", [4, 8])

    # Recurrent-specific
    gradient_steps = trial.suggest_categorical("gradient_steps", [8, 16])
    train_freq = 4  # fixed: 4:1 gradient-to-env-step ratio
    segment_len = trial.suggest_categorical("segment_len", [16, 32, 64])
    overlap_frac = trial.suggest_float("overlap_frac", 0.1, 0.4)
    overlap = max(1, int(segment_len * overlap_frac))
    burn_in = trial.suggest_categorical("burn_in", [0, 8])

    # Architecture
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128])
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
    net_arch_depth = trial.suggest_categorical("net_arch_depth", [2, 3])
    net_arch = [128] * net_arch_depth

    trial.set_user_attr("overlap", overlap)
    trial.set_user_attr("net_arch", net_arch)

    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        ent_coef=ent_coef,
        n_envs=n_envs,
        gradient_steps=gradient_steps,
        train_freq=train_freq,
        segment_len=segment_len,
        overlap=overlap,
        burn_in=burn_in,
        shared_state=True,  # single LSTM
        buffer_size=100_000,
        policy_kwargs=dict(
            net_arch=net_arch,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
        ),
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial, n_timesteps: int, seed: int = 0) -> float:
    params = sample_params(trial)
    n_envs = params.pop("n_envs")

    learning_starts = max(params["batch_size"] * 4, 1_000)

    try:
        train_env = make_env(n_envs=n_envs, seed=seed)
        eval_env = make_env(n_envs=1, seed=seed + 1000)

        model = RecurrentSAC(
            "MlpLstmPolicy",
            train_env,
            learning_starts=learning_starts,
            verbose=0,
            seed=seed,
            **params,
        )

        # Intermediate pruning: report reward at 1/3 and 2/3 of budget
        checkpoints = [n_timesteps // 3, 2 * n_timesteps // 3]
        prev_steps = 0
        for checkpoint in checkpoints:
            model.learn(total_timesteps=checkpoint - prev_steps, reset_num_timesteps=False)
            prev_steps = checkpoint

            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            eval_env.training = False
            eval_env.norm_reward = False

            interim_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=5, deterministic=True
            )
            eval_env.training = True
            eval_env.norm_reward = True

            trial.report(interim_reward, step=checkpoint)
            if trial.should_prune():
                train_env.close()
                eval_env.close()
                raise optuna.TrialPruned()

        # Final evaluation
        model.learn(total_timesteps=n_timesteps - prev_steps, reset_num_timesteps=False)

        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        train_env.close()
        eval_env.close()

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}", flush=True)
        return float("-inf")

    solved = "  *** SOLVED ***" if mean_reward >= SOLVED_THRESHOLD else ""
    print(
        f"  Trial {trial.number}: {mean_reward:.1f} ± {std_reward:.1f}{solved}",
        flush=True,
    )
    return mean_reward


# ---------------------------------------------------------------------------
# Study runner and results printer
# ---------------------------------------------------------------------------


def run_study(
    n_trials: int,
    n_timesteps: int,
    seed: int = 0,
    storage: str | None = None,
    n_jobs: int = 1,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="rsac_lunarlander_no_vel",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, n_timesteps, seed),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )
    return study


def print_results(study: optuna.Study) -> None:
    valid = sorted(
        [t for t in study.trials if t.value is not None and t.value > float("-inf")],
        key=lambda t: t.value,
        reverse=True,
    )
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.value == float("-inf")]

    print(f"\n{'=' * 65}")
    print("RecurrentSAC — LunarLanderContinuousNoVel-v3")
    print(f"{'=' * 65}")

    if not valid:
        print("No successful trials.")
        return

    best = study.best_trial
    print(f"Best mean reward : {best.value:.1f}")
    print(f"Solved (≥{SOLVED_THRESHOLD:.0f})  : {'yes' if best.value >= SOLVED_THRESHOLD else 'no'}")
    print("\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:25s}: {v}")
    for k, v in best.user_attrs.items():
        print(f"  {k:25s}: {v}  [derived]")

    print(f"\nTop-5 trials:")
    for t in valid[:5]:
        print(f"  Trial {t.number:3d}: {t.value:.1f}")

    print(f"\nTotal trials : {len(study.trials)}")
    print(f"  Successful : {len(valid)}")
    print(f"  Pruned     : {len(pruned)}")
    print(f"  Failed     : {len(failed)}")

    # Suggested config for training
    print(f"\n{'=' * 65}")
    print("Suggested training config (copy into your script):")
    print(f"{'=' * 65}")
    p = best.params
    ua = best.user_attrs
    net_arch = ua.get("net_arch", [256, 256])
    overlap = ua.get("overlap", p.get("segment_len", 32) // 4)
    print(
        f"""
model = RecurrentSAC(
    "MlpLstmPolicy",
    env,
    learning_rate={p['learning_rate']:.2e},
    gamma={p['gamma']},
    tau={p['tau']},
    batch_size={p['batch_size']},
    ent_coef={repr(p['ent_coef'])},
    train_freq=4,
    gradient_steps={p['gradient_steps']},
    segment_len={p['segment_len']},
    overlap={overlap},
    burn_in={p['burn_in']},
    shared_state=True,
    buffer_size=100_000,
    policy_kwargs={{
        "net_arch": {net_arch},
        "lstm_hidden_size": {p['lstm_hidden_size']},
        "n_lstm_layers": {p['n_lstm_layers']},
    }},
)"""
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune RecurrentSAC on LunarLanderContinuousNoVel")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=300_000,
        help="Training budget per trial (default: 300k)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///lunarlander.db (enables resuming)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel trials (requires shared storage)",
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Tuning RecurrentSAC on LunarLanderContinuousNoVel-v3", flush=True)
    print(f"  n_trials={args.n_trials}, n_timesteps={args.n_timesteps}, seed={args.seed}", flush=True)
    if args.storage:
        print(f"  storage={args.storage}", flush=True)

    study = run_study(
        n_trials=args.n_trials,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        storage=args.storage,
        n_jobs=args.n_jobs,
    )
    print_results(study)
