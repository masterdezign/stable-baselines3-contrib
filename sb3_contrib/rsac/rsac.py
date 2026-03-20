from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from sb3_contrib.rsac.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy, RecurrentSACPolicy
from sb3_contrib.rsac.replay_buffer import RecurrentReplayBuffer

SelfRecurrentSAC = TypeVar("SelfRecurrentSAC", bound="RecurrentSAC")


class RecurrentSAC(OffPolicyAlgorithm):
    """
    Recurrent Soft Actor-Critic (RSAC).

    An extension of SAC with LSTM-based actor and critic networks, enabling the agent to
    handle partially observable environments. Follows the R2D2 replay strategy: transitions
    are stored as fixed-length overlapping sequence chunks, and the LSTM is trained on full
    sequences with optional burn-in (the initial steps are used only for warming up the LSTM
    hidden state, not for gradient computation).

    References:
    - SAC: https://arxiv.org/abs/1801.01290
    - R2D2 (replay strategy): https://openreview.net/pdf?id=r1lyTjAqYX

    :param policy: Policy class or string alias (``"MlpLstmPolicy"``, ``"CnnLstmPolicy"``,
        ``"MultiInputLstmPolicy"``).
    :param env: Training environment.
    :param learning_rate: Learning rate for all optimizers.
    :param buffer_size: Number of sequence chunks in the replay buffer.
    :param learning_starts: Number of environment steps before training begins.
    :param batch_size: Number of chunks per gradient update.
    :param tau: Polyak update coefficient for the target critic.
    :param gamma: Discount factor.
    :param train_freq: How often to train (steps or episodes).
    :param gradient_steps: Gradient updates per training call.
    :param action_noise: Exploration noise added to actions.
    :param ent_coef: Entropy regularization coefficient (``"auto"`` to learn it).
    :param target_update_interval: Target network update frequency (gradient steps).
    :param target_entropy: Target entropy for automatic ``ent_coef`` tuning (``"auto"``
        sets it to ``-|action_dim|``).
    :param segment_len: Length of each stored sequence chunk (number of transitions).
    :param overlap: Number of transitions copied from the end of a chunk to the start
        of the next chunk. Provides context continuity across chunks.
    :param burn_in: Number of initial steps per chunk that are excluded from gradient
        computation. The LSTM uses these steps to warm up from the stored initial state.
    :param store_state: If True, the LSTM hidden state is stored in the replay buffer at
        the start of each chunk and used to initialize the LSTM during training.
        If False, the LSTM is always initialized from zeros (with burn-in for recovery).
    :param shared_state: If True, the critic reuses the actor LSTM output (detached).
        If False, the critic has its own independent LSTM.
    :param rnn_type: RNN cell type (currently only ``"LSTM"`` is supported).
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE).
    :param sde_sample_freq: Resample gSDE noise every N steps (-1 = only at rollout start).
    :param use_sde_at_warmup: Use gSDE during the warmup phase.
    :param stats_window_size: Window for episode statistics logging.
    :param tensorboard_log: TensorBoard log directory.
    :param policy_kwargs: Extra keyword arguments passed to the policy constructor.
    :param verbose: Verbosity level (0 = silent, 1 = info, 2 = debug).
    :param seed: Random seed.
    :param device: Torch device (``"auto"`` selects GPU if available).
    :param _init_setup_model: Whether to build the model in ``__init__``.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }
    policy: RecurrentSACPolicy

    def __init__(
        self,
        policy: str | type[RecurrentSACPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 10_000,
        learning_starts: int = 100,
        batch_size: int = 64,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        max_grad_norm: float = 10.0,
        segment_len: int = 50,
        overlap: int = 10,
        burn_in: int = 10,
        store_state: bool = True,
        shared_state: bool = True,
        rnn_type: str = "LSTM",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        if rnn_type != "LSTM":
            raise ValueError(f"rnn_type '{rnn_type}' is not supported. Only 'LSTM' is implemented.")

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            # Bypass the parent's replay buffer creation — we set up our own.
            replay_buffer_class=RecurrentReplayBuffer,
            replay_buffer_kwargs={},
            optimize_memory_usage=False,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.log_ent_coef: th.Tensor | None = None
        self.ent_coef_optimizer: th.optim.Adam | None = None

        self.segment_len = segment_len
        self.overlap = overlap
        self.burn_in = burn_in
        self.store_state = store_state
        self.shared_state = shared_state
        self.rnn_type = rnn_type

        # Persistent LSTM states per env for data collection
        self._last_lstm_states: tuple[th.Tensor, th.Tensor] | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Inject LSTM/shared_state settings into policy_kwargs
        if self.policy_kwargs is None:
            self.policy_kwargs: dict[str, Any] = {}
        self.policy_kwargs.setdefault("shared_state", self.shared_state)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        lstm_hidden_size = self.policy.lstm_hidden_size
        n_lstm_layers = self.policy.n_lstm_layers

        # Initialize per-env LSTM states to zero
        self._last_lstm_states = (
            th.zeros(n_lstm_layers, self.n_envs, lstm_hidden_size, device=self.device),
            th.zeros(n_lstm_layers, self.n_envs, lstm_hidden_size, device=self.device),
        )

        # Build the recurrent replay buffer
        self.replay_buffer = RecurrentReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            segment_len=self.segment_len,
            overlap=self.overlap,
            n_lstm_layers=n_lstm_layers,
            lstm_hidden_size=lstm_hidden_size,
            n_envs=self.n_envs,
            device=self.device,
        )

        self._convert_train_freq()

        # Entropy coefficient setup (same as SAC)
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore[union-attr]
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: RecurrentReplayBuffer,  # type: ignore[override]
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        """
        Collect environment steps and store them in the recurrent replay buffer.

        Maintains LSTM hidden states across steps, resetting them at episode boundaries.
        """
        assert self._last_lstm_states is not None
        assert self._last_obs is not None  # type: ignore[has-type]

        self.policy.set_training_mode(False)

        if isinstance(action_noise, VectorizedActionNoise):
            pass
        elif action_noise is not None and env.num_envs > 1:
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        num_collected_steps, num_collected_episodes = 0, 0
        continue_training = True

        lstm_states = deepcopy(self._last_lstm_states)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = self._obs_to_tensor(self._last_obs)  # type: ignore[has-type]
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)  # type: ignore[has-type]

                if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                    # Random warmup actions (still update LSTM state for continuity)
                    unscaled_actions = np.array([self.action_space.sample() for _ in range(env.num_envs)])
                    _, lstm_states = self.policy.actor(obs_tensor, lstm_states, episode_starts)
                else:
                    scaled_actions, lstm_states = self.policy.actor(obs_tensor, lstm_states, episode_starts)
                    unscaled_actions = self.policy.unscale_action(scaled_actions.cpu().numpy())

                    if action_noise is not None:
                        scaled_actions_np = scaled_actions.cpu().numpy()
                        scaled_actions_np = np.clip(scaled_actions_np + action_noise(), -1, 1)
                        unscaled_actions = self.policy.unscale_action(scaled_actions_np)

            # Scale actions for storage and stepping
            if isinstance(self.action_space, spaces.Box):
                buffer_actions = self.policy.scale_action(unscaled_actions)
                actions = unscaled_actions
            else:
                buffer_actions = unscaled_actions
                actions = unscaled_actions

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            self._update_info_buffer(infos, dones)

            # Store the transition (with terminal obs correction for VecEnv auto-reset)
            next_obs = self._get_next_obs(new_obs, dones, infos)

            replay_buffer.add(
                self._last_original_obs if self._vec_normalize_env is not None else self._last_obs,  # type: ignore[has-type]
                next_obs,
                buffer_actions,
                rewards if self._vec_normalize_env is None else self._vec_normalize_env.get_original_reward(),
                dones,
                infos,
                lstm_states=lstm_states if self.store_state else None,
            )

            self._last_obs = new_obs
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()
            self._last_episode_starts = dones

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # Reset LSTM state for envs that finished an episode
            for env_idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    lstm_states[0][:, env_idx] = 0.0
                    lstm_states[1][:, env_idx] = 0.0

                    if action_noise is not None:
                        kwargs = dict(indices=[env_idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()

        self._last_lstm_states = lstm_states
        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _get_next_obs(
        self,
        new_obs: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> np.ndarray:
        """Return the true next observations, correcting for VecEnv auto-reset on done."""
        from copy import deepcopy

        if self._vec_normalize_env is not None:
            next_obs = self._vec_normalize_env.get_original_obs()
        else:
            next_obs = deepcopy(new_obs)

        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                terminal = infos[i]["terminal_observation"]
                if self._vec_normalize_env is not None:
                    terminal = self._vec_normalize_env.unnormalize_obs(terminal)
                next_obs[i] = terminal

        return next_obs

    def _obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        """Convert a numpy obs array to a tensor on the policy device."""
        return th.tensor(obs, dtype=th.float32, device=self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Sample sequence chunks from the replay buffer and perform gradient updates.

        Each chunk of length ``segment_len`` is processed by the LSTM. The first
        ``burn_in`` steps are used only to warm up the LSTM state; losses are computed
        only on the remaining ``segment_len - burn_in`` steps.
        """
        if self.replay_buffer.size() < batch_size:
            return  # Not enough chunks in the buffer yet

        self.policy.set_training_mode(True)

        q_optimizers = [
            getattr(self.policy.critic, f"q_optimizer_{i}") for i in range(self.policy.n_critics)
        ]
        optimizers = [self.policy.actor.optimizer, *q_optimizers]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            B = batch_size
            T = self.segment_len
            burn_in = self.burn_in

            # Unpack: observations (B, T+1, obs_dim), etc.
            obs_all = replay_data.observations  # (B, T+1, obs_dim)
            acts = replay_data.actions  # (B, T, act_dim)
            rews = replay_data.rewards  # (B, T, 1)
            dones = replay_data.dones  # (B, T, 1)
            buffer_mask = replay_data.mask  # (B, T, 1)

            # Initial LSTM states for each chunk.
            # For the shared_state=True case, both actor and critic use the actor LSTM,
            # so stored states are meaningful.  For the non-shared case the critic has its
            # own LSTM (always zero-initialised), so we zero-initialise the actor as well
            # to keep both networks in a consistent state during training.
            if self.store_state and self.shared_state:
                h0 = replay_data.hidden_states.contiguous()  # (n_layers, B, hidden)
                c0 = replay_data.cell_states.contiguous()  # (n_layers, B, hidden)
            else:
                shape = replay_data.hidden_states.shape
                h0 = th.zeros(shape, device=self.device)
                c0 = th.zeros(shape, device=self.device)

            # episode_starts for each of the T+1 positions in the observation sequence:
            #   position 0     → 0 (LSTM continues from stored/zero state)
            #   position t ≥ 1 → dones[:, t-1] (episode boundary before this obs)
            ep_starts_all = th.zeros(B, T + 1, device=self.device)
            ep_starts_all[:, 1:] = dones[:, :, 0]
            ep_starts_all_flat = ep_starts_all.reshape(B * (T + 1))

            # Convenience slices reused for the non-shared critic
            ep_starts_curr_flat = ep_starts_all[:, :T].reshape(B * T)
            ep_starts_next_flat = ep_starts_all[:, 1:].reshape(B * T)
            obs_curr_flat = obs_all[:, :T].reshape(B * T, *obs_all.shape[2:])
            obs_next_flat = obs_all[:, 1:].reshape(B * T, *obs_all.shape[2:])

            # ── Single LSTM pass over all T+1 observations (matches the reference) ───
            # The actor LSTM sees [obs_0, obs_1, …, obs_T] in one forward pass.
            # Slicing gives consistent current and next latents on the same trajectory.
            obs_all_flat = obs_all.reshape(B * (T + 1), *obs_all.shape[2:])
            lstm_out_all, _ = self.policy.actor.get_lstm_latent(obs_all_flat, (h0, c0), ep_starts_all_flat)
            lstm_out_all_seq = lstm_out_all.reshape(B, T + 1, -1)
            lstm_curr_flat = lstm_out_all_seq[:, :T].reshape(B * T, -1)  # h after obs_t   (t=0..T-1)
            lstm_next_flat = lstm_out_all_seq[:, 1:].reshape(B * T, -1)  # h after obs_t+1 (t=1..T)

            # Compute entropy coefficient
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            if self.use_sde:
                self.policy.actor.reset_noise()

            # ── Actor MLP on current latent (for ent_coef and actor losses) ──────────
            curr_actions, log_prob = self.policy.actor.actions_log_prob_from_latent(lstm_curr_flat)
            log_prob = log_prob.reshape(B, T, 1)

            # ── Entropy coefficient update ─────────────────────────────────────────────
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Unmasked mean matches the reference: log_alpha * mean(entropy - target_entropy)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ── Target Q computation ───────────────────────────────────────────────────
            with th.no_grad():
                # Next actions from the already-computed next LSTM latent
                next_actions, next_log_prob = self.policy.actor.actions_log_prob_from_latent(lstm_next_flat)
                next_log_prob = next_log_prob.reshape(B, T, 1)

                next_acts_flat = next_actions  # (B*T, act_dim)

                if self.shared_state:
                    # Target critic takes raw actor LSTM output as state representation
                    next_q_values, _ = self.policy.critic_target(lstm_next_flat, next_acts_flat)
                else:
                    # Each Q in the target critic has its own LSTM; zero-init, burn-in recovers.
                    h0_c = th.zeros_like(h0)
                    c0_c = th.zeros_like(c0)
                    next_q_values, _ = self.policy.critic_target(
                        obs_next_flat, next_acts_flat, (h0_c, c0_c), ep_starts_next_flat
                    )

                next_q_values_cat = th.cat(next_q_values, dim=-1).reshape(B, T, -1)
                next_q_min, _ = th.min(next_q_values_cat, dim=-1, keepdim=True)  # (B, T, 1)
                target_q = rews + (1.0 - dones) * self.gamma * (next_q_min - ent_coef * next_log_prob)

            # ── Critic update ─────────────────────────────────────────────────────────
            acts_flat = acts.reshape(B * T, -1)

            if self.shared_state:
                # Critic receives actor LSTM latent (detached — no gradient into actor LSTM here)
                current_q_values, _ = self.policy.critic(lstm_curr_flat.detach(), acts_flat)
            else:
                h0_c = th.zeros_like(h0)
                c0_c = th.zeros_like(c0)
                current_q_values, _ = self.policy.critic(obs_curr_flat, acts_flat, (h0_c, c0_c), ep_starts_curr_flat)

            # Separate backward per Q-head — matches the reference's independent
            # Q1_summarizer_optimizer / Q1_optimizer / Q2_summarizer_optimizer / Q2_optimizer steps.
            current_q_flat_list = [q.reshape(B, T, 1) for q in current_q_values]
            critic_loss_total = 0.0
            for i, (q_pred, q_opt) in enumerate(zip(current_q_flat_list, q_optimizers)):
                q_loss = self._masked_mean((q_pred - target_q) ** 2, buffer_mask, burn_in)
                q_opt.zero_grad()
                # retain_graph for all but the last Q (features may be shared)
                q_loss.backward(retain_graph=(i < self.policy.n_critics - 1))
                nn.utils.clip_grad_norm_(self.policy.critic._q_params(i), self.max_grad_norm)
                q_opt.step()
                critic_loss_total += q_loss.item()
            critic_losses.append(critic_loss_total)

            # ── Actor update ──────────────────────────────────────────────────────────
            # lstm_curr_flat already has grad; no need to re-run the LSTM.
            curr_actions_flat = curr_actions  # (B*T, act_dim)

            if self.shared_state:
                # Critic takes actor LSTM state (detached) + new actions (with grad)
                q_pi_values, _ = self.policy.critic(lstm_curr_flat.detach(), curr_actions_flat)
            else:
                h0_c = th.zeros_like(h0)
                c0_c = th.zeros_like(c0)
                q_pi_values, _ = self.policy.critic(obs_curr_flat, curr_actions_flat, (h0_c, c0_c), ep_starts_curr_flat)

            q_pi_cat = th.cat(q_pi_values, dim=-1).reshape(B, T, -1)
            min_q_pi, _ = th.min(q_pi_cat, dim=-1, keepdim=True)  # (B, T, 1)

            actor_loss = self._masked_mean(ent_coef * log_prob - min_q_pi, buffer_mask, burn_in)
            actor_losses.append(actor_loss.item())

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.policy.actor.optimizer.step()

            # ── Target network update ─────────────────────────────────────────────────
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    @staticmethod
    def _masked_mean(tensor: th.Tensor, mask: th.Tensor, burn_in: int) -> th.Tensor:
        """
        Compute the mean of ``tensor`` over valid, non-burn-in steps.

        :param tensor: (B, T, 1) tensor of per-step values.
        :param mask: (B, T, 1) buffer mask (1 = valid transition, 0 = padding).
        :param burn_in: Number of leading steps to exclude from the loss.
        :return: Scalar mean over valid non-burn-in steps.
        """
        B, T, _ = tensor.shape
        # Burn-in mask: zeros for first burn_in steps
        burn_mask = th.ones(B, T, 1, device=tensor.device)
        if burn_in > 0:
            burn_mask[:, :burn_in] = 0.0
        combined = mask * burn_mask
        denom = combined.sum().clamp(min=1.0)
        return (tensor * combined).sum() / denom

    def learn(
        self: SelfRecurrentSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "RecurrentSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.actor.optimizer"]
        for i in range(self.policy.n_critics):
            state_dicts.append(f"policy.critic.q_optimizer_{i}")
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            return state_dicts, ["log_ent_coef"]
        return state_dicts, ["ent_coef_tensor"]
