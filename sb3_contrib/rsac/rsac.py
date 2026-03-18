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

        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
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

            # Initial LSTM states for each chunk (from stored or zeros if not store_state)
            if self.store_state:
                h0 = replay_data.hidden_states.contiguous()  # (n_layers, B, hidden)
                c0 = replay_data.cell_states.contiguous()  # (n_layers, B, hidden)
            else:
                shape = replay_data.hidden_states.shape
                h0 = th.zeros(shape, device=self.device)
                c0 = th.zeros(shape, device=self.device)

            # episode_starts for current obs [0..T-1]: reset LSTM at episode boundaries
            # episode_starts[:, 0] = 0 (continue from stored state)
            # episode_starts[:, t] = dones[:, t-1] for t > 0
            ep_starts_curr = th.zeros(B, T, device=self.device)
            ep_starts_curr[:, 1:] = dones[:, :-1, 0]
            ep_starts_curr_flat = ep_starts_curr.reshape(B * T)  # (B*T,)

            # episode_starts for next obs [1..T]: dones[:, t] signals reset before next obs t
            ep_starts_next = dones[:, :, 0].reshape(B * T)  # (B*T,)

            # Flatten obs for LSTM processing
            obs_curr_flat = obs_all[:, :T].reshape(B * T, *obs_all.shape[2:])  # (B*T, obs_dim)
            obs_next_flat = obs_all[:, 1:].reshape(B * T, *obs_all.shape[2:])  # (B*T, obs_dim)

            # Compute entropy coefficient
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            if self.use_sde:
                self.policy.actor.reset_noise()

            # ── Actor forward on current obs (for ent_coef and actor losses) ──────────
            curr_actions, log_prob, _ = self.policy.actor.action_log_prob(obs_curr_flat, (h0, c0), ep_starts_curr_flat)
            log_prob = log_prob.reshape(B, T, 1)

            # ── Entropy coefficient update ─────────────────────────────────────────────
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach())
                ent_coef_loss = self._masked_mean(ent_coef_loss, buffer_mask, burn_in)
                ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ── Target Q computation ───────────────────────────────────────────────────
            with th.no_grad():
                # Actor on next obs (same initial state approximation; burn-in corrects transient error)
                next_actions, next_log_prob, _ = self.policy.actor.action_log_prob(obs_next_flat, (h0, c0), ep_starts_next)
                next_log_prob = next_log_prob.reshape(B, T, 1)

                next_acts_flat = next_actions  # (B*T, act_dim)

                if self.shared_state:
                    # Critic uses raw actor LSTM output (before actor MLP) as state representation.
                    # Re-run actor LSTM on next obs to get the raw LSTM latent.
                    from sb3_contrib.rsac.policies import _process_sequence

                    next_features = self.policy.actor.extract_features(obs_next_flat, self.policy.actor.features_extractor)
                    next_latent, _ = _process_sequence(next_features, (h0, c0), ep_starts_next, self.policy.actor.lstm)
                    # Target critic: feedforward from raw LSTM output
                    next_q_values, _ = self.policy.critic_target(next_latent, next_acts_flat)
                else:
                    # Target critic has its own LSTM; initialize from zeros (burn-in recovers)
                    h0_c = th.zeros_like(h0)
                    c0_c = th.zeros_like(c0)
                    next_q_values, _ = self.policy.critic_target(obs_next_flat, next_acts_flat, (h0_c, c0_c), ep_starts_next)

                next_q_values_cat = th.cat(next_q_values, dim=-1).reshape(B, T, -1)
                next_q_min, _ = th.min(next_q_values_cat, dim=-1, keepdim=True)  # (B, T, 1)
                target_q = rews + (1.0 - dones) * self.gamma * (next_q_min - ent_coef * next_log_prob)

            # ── Critic update ─────────────────────────────────────────────────────────
            acts_flat = acts.reshape(B * T, -1)

            if self.shared_state:
                # Critic uses raw actor LSTM output (detached) as state representation.
                from sb3_contrib.rsac.policies import _process_sequence

                with th.no_grad():
                    curr_features = self.policy.actor.extract_features(obs_curr_flat, self.policy.actor.features_extractor)
                    curr_latent, _ = _process_sequence(curr_features, (h0, c0), ep_starts_curr_flat, self.policy.actor.lstm)
                current_q_values, _ = self.policy.critic(curr_latent.detach(), acts_flat)
            else:
                h0_c = th.zeros_like(h0)
                c0_c = th.zeros_like(c0)
                current_q_values, _ = self.policy.critic(obs_curr_flat, acts_flat, (h0_c, c0_c), ep_starts_curr_flat)

            current_q_flat_list = [q.reshape(B, T, 1) for q in current_q_values]
            critic_loss = sum(
                self._masked_mean((q - target_q) ** 2 * 0.5, buffer_mask, burn_in)  # type: ignore[arg-type]
                for q in current_q_flat_list
            )
            critic_losses.append(critic_loss.item())

            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # ── Actor update ──────────────────────────────────────────────────────────
            curr_actions_flat = curr_actions  # (B*T, act_dim)

            if self.shared_state:
                # Actor loss: run actor LSTM with gradient; critic uses the raw LSTM latent
                # (detached from state path) plus the new actions (with gradient).
                from sb3_contrib.rsac.policies import _process_sequence

                curr_features_grad = self.policy.actor.extract_features(obs_curr_flat, self.policy.actor.features_extractor)
                curr_latent_grad, _ = _process_sequence(
                    curr_features_grad, (h0, c0), ep_starts_curr_flat, self.policy.actor.lstm
                )
                # Critic receives the state (detached) + new actions (with gradient)
                q_pi_values, _ = self.policy.critic(curr_latent_grad.detach(), curr_actions_flat)
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
        state_dicts = ["policy", "policy.actor.optimizer", "policy.critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            return state_dicts, ["log_ent_coef"]
        return state_dicts, ["ent_coef_tensor"]
