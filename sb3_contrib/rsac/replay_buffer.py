from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


class RecurrentReplayBufferSamples(NamedTuple):
    observations: th.Tensor  # (B, T+1, *obs_shape): obs at t=0..T-1 and next_obs at t=T-1
    actions: th.Tensor  # (B, T, act_dim)
    rewards: th.Tensor  # (B, T, 1)
    dones: th.Tensor  # (B, T, 1)
    mask: th.Tensor  # (B, T, 1): 1=valid step, 0=padding
    hidden_states: th.Tensor  # (n_layers, B, hidden_size): initial LSTM hidden state per chunk
    cell_states: th.Tensor  # (n_layers, B, hidden_size): initial LSTM cell state per chunk


class RecurrentReplayBuffer(BaseBuffer):
    """
    Replay buffer for recurrent off-policy algorithms (R2D2-style).

    Stores fixed-length sequence chunks with overlap between consecutive chunks from the
    same episode. Optionally stores the LSTM state at the start of each chunk to avoid
    relying entirely on zero-start initialization.

    The ``buffer_size`` parameter refers to the number of chunks (sequences), not individual
    transitions. Each chunk has ``segment_len`` steps, and ``overlap`` steps are copied from
    the end of one chunk to the start of the next.

    :param buffer_size: Number of chunks to store.
    :param observation_space: Observation space.
    :param action_space: Action space.
    :param segment_len: Number of transitions per chunk.
    :param overlap: Number of steps to copy from end of chunk to start of next chunk.
    :param n_lstm_layers: Number of LSTM layers (used for stored initial states).
    :param lstm_hidden_size: LSTM hidden size (used for stored initial states).
    :param n_envs: Number of parallel environments.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        segment_len: int = 50,
        overlap: int = 10,
        n_lstm_layers: int = 1,
        lstm_hidden_size: int = 256,
        n_envs: int = 1,
        device: th.device | str = "auto",
        **kwargs: Any,
    ):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)
        assert isinstance(self.obs_shape, tuple), "RecurrentReplayBuffer does not support dict observations"
        assert overlap < segment_len, f"overlap ({overlap}) must be less than segment_len ({segment_len})"
        self.segment_len = segment_len
        self.overlap = overlap
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.reset()

    def reset(self) -> None:
        # observations[chunk, t] = obs at step t; observations[chunk, t+1] = next_obs at step t
        self.observations = np.zeros((self.buffer_size, self.segment_len + 1, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.segment_len, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.segment_len, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.segment_len, 1), dtype=np.float32)
        # 1 = valid transition, 0 = padding (no data)
        self.mask = np.zeros((self.buffer_size, self.segment_len, 1), dtype=np.float32)

        # Stored LSTM state at start of each chunk: (buffer_size, n_layers, hidden)
        self.hidden_states = np.zeros((self.buffer_size, self.n_lstm_layers, self.lstm_hidden_size), dtype=np.float32)
        self.cell_states = np.zeros((self.buffer_size, self.n_lstm_layers, self.lstm_hidden_size), dtype=np.float32)

        # Per-env tracking
        self.time_pos = np.zeros(self.n_envs, dtype=np.int64)  # time position within current chunk
        self.current_chunk = np.arange(self.n_envs, dtype=np.int64)  # buffer slot for each env

        # Pending LSTM state: recorded at the overlap start point, used as initial state for next chunk
        self._pending_hidden = np.zeros((self.n_envs, self.n_lstm_layers, self.lstm_hidden_size), dtype=np.float32)
        self._pending_cell = np.zeros((self.n_envs, self.n_lstm_layers, self.lstm_hidden_size), dtype=np.float32)

        # _write_pos: next chunk slot to allocate (unbounded, use % buffer_size)
        # Initial n_envs slots are pre-allocated; the first committed chunk claims slot n_envs.
        self._write_pos = self.n_envs
        # self.pos tracks committed chunk count (used by size())
        self.pos = 0
        self.full = False

    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        lstm_states: tuple[th.Tensor, th.Tensor] | None = None,
    ) -> None:
        """
        Add a transition for each environment.

        :param obs: Current observations, shape (n_envs, *obs_shape).
        :param next_obs: Next observations, shape (n_envs, *obs_shape).
        :param action: Actions taken, shape (n_envs, act_dim).
        :param reward: Rewards received, shape (n_envs,).
        :param done: Episode termination flags, shape (n_envs,).
        :param infos: Per-env info dicts.
        :param lstm_states: Optional LSTM state tuple (hidden, cell), each (n_layers, n_envs, hidden).
            Used to store the initial state for new chunks.
        """
        for env_idx in range(self.n_envs):
            chunk_idx = self.current_chunk[env_idx]
            t = int(self.time_pos[env_idx])

            self.observations[chunk_idx, t] = np.array(obs[env_idx])
            self.observations[chunk_idx, t + 1] = np.array(next_obs[env_idx])
            self.actions[chunk_idx, t] = np.array(action[env_idx])
            self.rewards[chunk_idx, t] = float(reward[env_idx])
            self.dones[chunk_idx, t] = float(done[env_idx])
            self.mask[chunk_idx, t] = 1.0

            # At the overlap boundary, record the current LSTM state.
            # This state will be used as the initial state for the next chunk (overlap region).
            overlap_start = self.segment_len - self.overlap - 1
            if lstm_states is not None and t == overlap_start:
                self._pending_hidden[env_idx] = lstm_states[0][:, env_idx].detach().cpu().numpy()
                self._pending_cell[env_idx] = lstm_states[1][:, env_idx].detach().cpu().numpy()

            self.time_pos[env_idx] += 1
            end_of_chunk = int(self.time_pos[env_idx]) == self.segment_len
            episode_done = bool(done[env_idx])

            if end_of_chunk or episode_done:
                # Commit the current chunk
                self.pos = min(self.pos + 1, self.buffer_size)
                if self.pos >= self.buffer_size:
                    self.full = True

                # Allocate new chunk slot
                new_chunk_idx = self._write_pos % self.buffer_size
                self._write_pos += 1

                # Clear the new slot
                self.mask[new_chunk_idx] = 0.0
                self.observations[new_chunk_idx] = 0.0
                self.hidden_states[new_chunk_idx] = 0.0
                self.cell_states[new_chunk_idx] = 0.0

                if end_of_chunk and not episode_done:
                    # Copy last `overlap` steps to the start of the new chunk
                    ov = self.overlap
                    self.observations[new_chunk_idx, : ov + 1] = self.observations[chunk_idx, -(ov + 1) :]
                    self.actions[new_chunk_idx, :ov] = self.actions[chunk_idx, -ov:]
                    self.rewards[new_chunk_idx, :ov] = self.rewards[chunk_idx, -ov:]
                    self.dones[new_chunk_idx, :ov] = self.dones[chunk_idx, -ov:]
                    self.mask[new_chunk_idx, :ov] = 1.0

                    # Initial LSTM state for new chunk = state recorded at overlap start
                    self.hidden_states[new_chunk_idx] = self._pending_hidden[env_idx]
                    self.cell_states[new_chunk_idx] = self._pending_cell[env_idx]

                    self.time_pos[env_idx] = ov
                else:
                    # Episode ended: new chunk starts fresh (zeros already set above)
                    self.time_pos[env_idx] = 0

                self.current_chunk[env_idx] = new_chunk_idx

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> RecurrentReplayBufferSamples:
        obs = self.observations[batch_inds]  # (B, T+1, *obs_shape)
        if env is not None:
            # Normalize: collapse batch dims, normalize, reshape back
            B, Tp1 = obs.shape[:2]
            obs = self._normalize_obs(obs.reshape(B * Tp1, *self.obs_shape), env)
            obs = obs.reshape(B, Tp1, *self.obs_shape)

        actions = self.actions[batch_inds]  # (B, T, act_dim)
        rewards = self.rewards[batch_inds]  # (B, T, 1)
        dones = self.dones[batch_inds]  # (B, T, 1)
        mask = self.mask[batch_inds]  # (B, T, 1)

        # LSTM states: (B, n_layers, hidden) -> (n_layers, B, hidden)
        hidden = self.hidden_states[batch_inds].swapaxes(0, 1)
        cell = self.cell_states[batch_inds].swapaxes(0, 1)

        data = (obs, actions, rewards, dones, mask, hidden, cell)
        return RecurrentReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> RecurrentReplayBufferSamples:
        """Sample a batch of chunks."""
        upper = self.buffer_size if self.full else self.pos
        if upper == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        batch_inds = np.random.randint(0, upper, size=batch_size)
        return self._get_samples(batch_inds, env=env)
