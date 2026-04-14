"""Rolling observation history wrapper for LiT-style temporal Transformer.

Wraps a gym.Env so that each observation is a window of the last K timesteps
instead of just the latest snapshot. Must live at the env level (not the agent)
so that vectorized environments each maintain their own independent history.
"""

import numpy as np
import gymnasium as gym


class HistoryWrapper(gym.Wrapper):
    """Rolling observation history buffer.

    Output observation shape: (history_len, original_obs_dim)
      - buffer[0] = oldest observation
      - buffer[-1] = newest observation

    At episode reset, the buffer is filled with K copies of the reset observation
    so the shape is always (K, obs_dim) regardless of how far into the episode
    we are. This avoids variable-length inputs but means the first K-1 steps
    contain redundant information.
    """

    def __init__(self, env: gym.Env, history_len: int = 8):
        super().__init__(env)
        assert history_len >= 1, f"history_len must be >= 1, got {history_len}"

        base_space = env.observation_space
        assert len(base_space.shape) == 1, (
            f"HistoryWrapper only supports flat observation spaces, "
            f"got shape {base_space.shape}"
        )
        obs_dim = base_space.shape[0]

        self.history_len = history_len
        self.obs_dim = obs_dim
        self.buffer = np.zeros((history_len, obs_dim), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(history_len, obs_dim),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_f32 = np.asarray(obs, dtype=np.float32)
        self.buffer = np.tile(obs_f32[np.newaxis, :], (self.history_len, 1))
        return self.buffer.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_f32 = np.asarray(obs, dtype=np.float32)
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs_f32
        return self.buffer.copy(), reward, terminated, truncated, info
