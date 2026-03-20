from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def _process_sequence(
    features: th.Tensor,
    lstm_states: tuple[th.Tensor, th.Tensor],
    episode_starts: th.Tensor,
    lstm: nn.LSTM,
) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    """
    Run a sequence of features through an LSTM, resetting hidden states at episode boundaries.

    :param features: (n_seq * seq_len, features_dim) — flat batch; n_seq inferred from lstm_states.
    :param lstm_states: (h, c) each (n_layers, n_seq, hidden_size).
    :param episode_starts: (n_seq * seq_len,) — 1.0 at positions where a new episode begins.
    :param lstm: LSTM module.
    :return: (output, new_lstm_states) where output has shape (n_seq * seq_len, hidden_size).
    """
    n_seq = lstm_states[0].shape[1]
    # Reshape: (n_seq * seq_len, feat) -> (n_seq, seq_len, feat) -> (seq_len, n_seq, feat)
    features_seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    ep_starts_seq = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    if th.all(ep_starts_seq == 0.0):
        lstm_out, lstm_states = lstm(features_seq, lstm_states)
        # (seq_len, n_seq, hidden) -> (n_seq, seq_len, hidden) -> (n_seq * seq_len, hidden)
        lstm_out = th.flatten(lstm_out.transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_out, lstm_states

    lstm_out = []
    for feat_t, ep_start_t in zip(features_seq, ep_starts_seq, strict=True):
        hidden, lstm_states = lstm(
            feat_t.unsqueeze(0),
            (
                (1.0 - ep_start_t).view(1, n_seq, 1) * lstm_states[0],
                (1.0 - ep_start_t).view(1, n_seq, 1) * lstm_states[1],
            ),
        )
        lstm_out.append(hidden)
    lstm_out = th.flatten(th.cat(lstm_out).transpose(0, 1), start_dim=0, end_dim=1)
    return lstm_out, lstm_states


class RecurrentActor(BasePolicy):
    """
    Actor (policy) network for RecurrentSAC, with an LSTM layer.

    Architecture: features_extractor → LSTM → MLP → (mean, log_std).

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param net_arch: MLP hidden layer sizes applied after the LSTM.
    :param features_extractor: Feature extraction network.
    :param features_dim: Output dimension of the features extractor.
    :param lstm_hidden_size: Number of units in the LSTM hidden state.
    :param n_lstm_layers: Number of stacked LSTM layers.
    :param activation_fn: Activation function for the MLP.
    :param use_sde: Whether to use State Dependent Exploration (gSDE).
    :param log_std_init: Initial log standard deviation for the action distribution.
    :param full_std: Use full covariance for gSDE.
    :param use_expln: Use expln() instead of exp() for gSDE.
    :param clip_mean: Clip the mean output of gSDE.
    :param normalize_images: Whether to normalize image observations.
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.features_dim = features_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        self.clip_mean = clip_mean

        self.lstm = nn.LSTM(features_dim, lstm_hidden_size, num_layers=n_lstm_layers)
        # Shape used for initializing hidden states in predict(): (n_layers, 1, hidden)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)

        # MLP applied to LSTM output before mean/log_std heads
        latent_net = create_mlp(lstm_hidden_size, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_net)
        last_dim = net_arch[-1] if net_arch else lstm_hidden_size

        action_dim = get_action_dim(action_space)
        if use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_dim, latent_sde_dim=last_dim, log_std_init=log_std_init
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_dim, action_dim)
            self.log_std = nn.Linear(last_dim, action_dim)  # type: ignore[assignment]

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                lstm_hidden_size=self.lstm_hidden_size,
                n_lstm_layers=self.n_lstm_layers,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """Return the gSDE standard deviation (only when use_sde=True)."""
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "get_std() requires use_sde=True"
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE noise weights (only when use_sde=True)."""
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() requires use_sde=True"
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor], tuple[th.Tensor, th.Tensor]]:
        """
        Compute action distribution parameters for a (possibly batched) sequence of observations.

        :param obs: Observations of shape (n_seq * seq_len, *obs_shape).
        :param lstm_states: (h, c) each (n_layers, n_seq, hidden_size).
        :param episode_starts: (n_seq * seq_len,) with 1.0 at episode boundaries.
        :return: (mean, log_std, kwargs, new_lstm_states).
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, new_lstm_states = _process_sequence(features, lstm_states, episode_starts, self.lstm)
        latent_pi = self.latent_pi(latent_pi)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi), new_lstm_states
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}, new_lstm_states

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        mean, log_std, kwargs, new_states = self.get_action_dist_params(obs, lstm_states, episode_starts)
        actions = self.action_dist.actions_from_params(mean, log_std, deterministic=deterministic, **kwargs)
        return actions, new_states

    def action_log_prob(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, th.Tensor]]:
        mean, log_std, kwargs, new_states = self.get_action_dist_params(obs, lstm_states, episode_starts)
        actions, log_prob = self.action_dist.log_prob_from_params(mean, log_std, **kwargs)
        return actions, log_prob, new_states

    def get_lstm_latent(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Run features extractor + LSTM only; return raw LSTM output before the MLP heads."""
        features = self.extract_features(obs, self.features_extractor)
        lstm_out, new_states = _process_sequence(features, lstm_states, episode_starts, self.lstm)
        return lstm_out, new_states

    def actions_log_prob_from_latent(
        self,
        lstm_out: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Apply MLP heads to precomputed LSTM output; return (squashed actions, log_prob)."""
        latent_pi = self.latent_pi(lstm_out)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return self.action_dist.log_prob_from_params(mean_actions, self.log_std, latent_sde=latent_pi)
        log_std = th.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)  # type: ignore[operator]
        return self.action_dist.log_prob_from_params(mean_actions, log_std)

    def _predict(
        self,
        observation: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        return self.forward(observation, lstm_states, episode_starts, deterministic)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, np.ndarray] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Get actions given observations, managing LSTM states as numpy arrays.

        :param observation: Input observation(s).
        :param state: Optional (h, c) LSTM state, each (n_layers, n_envs, hidden_size).
        :param episode_start: (n_envs,) — 1 at start of each episode.
        :param deterministic: If True, take the mean action.
        :return: (actions, new_state).
        """
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)
        n_envs = observation.shape[0] if not isinstance(observation, dict) else next(iter(observation.values())).shape[0]

        if state is None:
            h = np.zeros((self.n_lstm_layers, n_envs, self.lstm_hidden_size), dtype=np.float32)
            state = (h, h.copy())
        if episode_start is None:
            episode_start = np.zeros(n_envs, dtype=np.float32)

        with th.no_grad():
            lstm_states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, new_lstm_states = self._predict(observation, lstm_states, episode_starts, deterministic)
            new_state = (new_lstm_states[0].cpu().numpy(), new_lstm_states[1].cpu().numpy())

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[assignment]
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, new_state


class RecurrentCritic(BaseModel):
    """
    Critic (Q-value) network for RecurrentSAC.

    When ``shared_state=False``, the critic has its own LSTM and independently processes
    observations. When ``shared_state=True``, the critic is a feedforward network that
    receives the pre-computed actor LSTM output as input (no separate LSTM).

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param net_arch: MLP hidden layer sizes after the LSTM (or after input if shared_state).
    :param features_extractor: Feature extraction network (ignored when shared_state=True).
    :param features_dim: Output dimension of the features extractor.
    :param lstm_hidden_size: LSTM hidden size.
    :param n_lstm_layers: Number of LSTM layers.
    :param activation_fn: MLP activation function.
    :param normalize_images: Whether to normalize image observations.
    :param n_critics: Number of Q-network heads.
    :param shared_state: If True, actor LSTM output is passed directly (no critic LSTM).
    """

    action_space: spaces.Box
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        shared_state: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(action_space)
        self.shared_state = shared_state
        self.n_critics = n_critics
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers

        # When shared_state=False each Q-head gets its own independent LSTM (matching the
        # reference offpcc implementation where Q1_summarizer and Q2_summarizer are separate).
        if not shared_state:
            self.lstm_list = nn.ModuleList(
                [nn.LSTM(features_dim, lstm_hidden_size, num_layers=n_lstm_layers) for _ in range(n_critics)]
            )
        else:
            self.lstm_list = None  # type: ignore[assignment]

        critic_input_dim = lstm_hidden_size + action_dim
        self.q_networks: list[nn.Module] = []
        for i in range(n_critics):
            qf = nn.Sequential(*create_mlp(critic_input_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{i}", qf)
            self.q_networks.append(qf)

    def forward(
        self,
        obs_or_latent: th.Tensor,
        actions: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor] | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[tuple[th.Tensor, ...], tuple[th.Tensor, th.Tensor] | None]:
        """
        Compute Q-values.

        :param obs_or_latent: Observations (n_seq*T, *obs_shape) when not shared_state;
            actor LSTM output (n_seq*T, hidden) when shared_state.
        :param actions: (n_seq*T, act_dim).
        :param lstm_states: (h, c) each (n_layers, n_seq, hidden) — required when not shared_state.
        :param episode_starts: (n_seq*T,) — required when not shared_state.
        :return: (tuple of Q-value tensors each (n_seq*T, 1), new_lstm_states or None).
        """
        if self.lstm_list is not None and lstm_states is not None and episode_starts is not None:
            # Non-shared case: each Q-head runs through its own independent LSTM.
            features = self.extract_features(obs_or_latent, self.features_extractor)
            q_values = tuple(
                qf(th.cat([_process_sequence(features, lstm_states, episode_starts, lstm_i)[0], actions], dim=-1))
                for qf, lstm_i in zip(self.q_networks, self.lstm_list)
            )
            return q_values, None
        else:
            # Shared-state case: actor LSTM output passed directly as latent.
            qvalue_input = th.cat([obs_or_latent, actions], dim=-1)
            q_values = tuple(qf(qvalue_input) for qf in self.q_networks)
            return q_values, lstm_states


class RecurrentSACPolicy(BasePolicy):
    """
    Policy class for RecurrentSAC.

    Combines a recurrent actor (LSTM + MLP → action) with a recurrent or feedforward
    critic (optional LSTM + MLP → Q-values) and its target network.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param lr_schedule: Learning rate schedule.
    :param net_arch: Network architecture — list of ints (shared pi/vf) or dict with
        ``pi`` and ``qf`` keys.
    :param activation_fn: Activation function.
    :param use_sde: Whether to use gSDE exploration.
    :param log_std_init: Initial log standard deviation.
    :param use_expln: Use expln() for gSDE.
    :param clip_mean: Clip actor mean output for gSDE.
    :param features_extractor_class: Features extractor class.
    :param features_extractor_kwargs: Keyword arguments for the features extractor.
    :param normalize_images: Whether to normalize image observations.
    :param optimizer_class: Optimizer class.
    :param optimizer_kwargs: Optimizer keyword arguments (excluding learning rate).
    :param n_critics: Number of Q-network heads.
    :param shared_state: Whether the critic reuses the actor LSTM output (True) or has
        its own LSTM (False).
    :param lstm_hidden_size: LSTM hidden state dimension.
    :param n_lstm_layers: Number of stacked LSTM layers.
    """

    actor: RecurrentActor
    critic: RecurrentCritic
    critic_target: RecurrentCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        shared_state: bool = True,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.shared_state = shared_state
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.n_critics = n_critics

        self.net_args: dict[str, Any] = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": activation_fn,
            "normalize_images": normalize_images,
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
        }
        self.actor_kwargs = {
            **self.net_args,
            "net_arch": actor_arch,
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.critic_kwargs = {
            **self.net_args,
            "net_arch": critic_arch,
            "n_critics": n_critics,
            "shared_state": shared_state,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Critic has its own separate features extractor
        self.critic = self.make_critic(features_extractor=None)
        self.critic.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.critic.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                n_critics=self.n_critics,
                shared_state=self.shared_state,
                lstm_hidden_size=self.lstm_hidden_size,
                n_lstm_layers=self.n_lstm_layers,
            )
        )
        return data

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> RecurrentActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return RecurrentActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> RecurrentCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return RecurrentCritic(**critic_kwargs).to(self.device)

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE noise weights (only when use_sde=True)."""
        self.actor.reset_noise(batch_size=batch_size)

    def forward(
        self,
        obs: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        return self._predict(obs, lstm_states, episode_starts, deterministic)

    def _predict(
        self,
        observation: PyTorchObs,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        return self.actor(observation, lstm_states, episode_starts, deterministic)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, np.ndarray] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Get policy actions, managing LSTM states as numpy arrays.

        :param observation: Input observation(s).
        :param state: Optional (h, c) LSTM state, each (n_layers, n_envs, hidden_size).
        :param episode_start: (n_envs,) — 1 at episode starts.
        :param deterministic: If True, take the mean action.
        :return: (actions, new_state).
        """
        return self.actor.predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpLstmPolicy = RecurrentSACPolicy


class CnnLstmPolicy(RecurrentSACPolicy):
    """RecurrentSACPolicy with CNN features extractor."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        shared_state: bool = True,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            shared_state,
            lstm_hidden_size,
            n_lstm_layers,
        )


class MultiInputLstmPolicy(RecurrentSACPolicy):
    """RecurrentSACPolicy with CombinedExtractor for dict observations."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        shared_state: bool = True,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            shared_state,
            lstm_hidden_size,
            n_lstm_layers,
        )
