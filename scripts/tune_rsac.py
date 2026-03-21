"""
Optuna hyperparameter search for RecurrentSAC on POMDP benchmarks.

Environments (NoVel = velocity observations removed, making them POMDPs):
  - PendulumNoVel-v1
  - MountainCarContinuousNoVel-v0
  - LunarLanderContinuousNoVel-v3  (requires Box2D / swig)

Usage:
    python scripts/tune_rsac.py --env PendulumNoVel-v1 --n-trials 25 --n-timesteps 25000
    python scripts/tune_rsac.py --env MountainCarContinuousNoVel-v0 --n-trials 20 --n-timesteps 40000
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import optuna
import rl_zoo3.import_envs  # noqa: F401 — registers NoVel envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentSAC

N_EVAL_EPISODES = 10


def make_env(env_id: str, n_envs: int, seed: int = 0) -> VecNormalize:
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    return VecNormalize(env, norm_obs=True, norm_reward=True)


def sample_params(trial: optuna.Trial) -> dict:
    """Sample RecurrentSAC hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True)
    gamma = trial.suggest_categorical("gamma", [0.95, 0.99])
    tau = trial.suggest_categorical("tau", [0.01, 0.02])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.01])
    n_envs = trial.suggest_categorical("n_envs", [2, 4])

    gradient_steps = trial.suggest_categorical("gradient_steps", [4, 8, 16])

    # Recurrent-specific
    segment_len = trial.suggest_categorical("segment_len", [10, 20, 32])
    overlap_frac = trial.suggest_float("overlap_frac", 0.1, 0.35)
    overlap = max(1, int(segment_len * overlap_frac))
    burn_in = trial.suggest_categorical("burn_in", [0, 5])
    shared_state = trial.suggest_categorical("shared_state", [True])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64])

    trial.set_user_attr("overlap", overlap)

    return dict(
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        ent_coef=ent_coef,
        n_envs=n_envs,
        gradient_steps=gradient_steps,
        train_freq=4,
        segment_len=segment_len,
        overlap=overlap,
        burn_in=burn_in,
        shared_state=shared_state,
        policy_kwargs=dict(net_arch=[64], lstm_hidden_size=lstm_hidden_size),
    )


def objective(trial: optuna.Trial, env_id: str, n_timesteps: int, seed: int = 0) -> float:
    params = sample_params(trial)
    n_envs = params.pop("n_envs")

    segment_len = params["segment_len"]
    buffer_size = max(512, 30_000 // segment_len)
    learning_starts = max(params["batch_size"] * 2, 256)

    try:
        train_env = make_env(env_id, n_envs=n_envs, seed=seed)
        eval_env = make_env(env_id, n_envs=1, seed=seed + 1000)

        model = RecurrentSAC(
            "MlpLstmPolicy",
            train_env,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            verbose=0,
            seed=seed,
            **params,
        )
        model.learn(total_timesteps=n_timesteps)

        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        train_env.close()
        eval_env.close()
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}", flush=True)
        return float("-inf")

    print(f"  Trial {trial.number}: mean_reward={mean_reward:.2f} ± {std_reward:.2f}", flush=True)
    return mean_reward


def run_study(env_id: str, n_trials: int, n_timesteps: int, seed: int = 0) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"rsac_{env_id}",
    )
    study.optimize(
        lambda trial: objective(trial, env_id, n_timesteps, seed),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def print_results(study: optuna.Study, env_id: str) -> None:
    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"Best result for {env_id}")
    print(f"{'=' * 60}")
    print(f"  Value (mean reward): {best.value:.2f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    for k, v in best.user_attrs.items():
        print(f"    {k}: {v}  [derived]")

    valid = sorted(
        [t for t in study.trials if t.value is not None and t.value > float("-inf")],
        key=lambda t: t.value,
        reverse=True,
    )
    print(f"\n  Top-5 trials:")
    for t in valid[:5]:
        print(f"    Trial {t.number:3d}: {t.value:.2f}")
    print(f"  Failed / pruned: {len(study.trials) - len(valid)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PendulumNoVel-v1")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-timesteps", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Tuning RecurrentSAC on {args.env}", flush=True)
    print(f"  n_trials={args.n_trials}, n_timesteps={args.n_timesteps}, seed={args.seed}", flush=True)

    study = run_study(args.env, args.n_trials, args.n_timesteps, args.seed)
    print_results(study, args.env)
