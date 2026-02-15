import numpy as np
import gymnasium as gym

# *** wrappers no longer used due to being extremely slow ***
class ActionDelayWrapper(gym.Wrapper):
    """
    Adds fixed-step delays for each component of the action vector.
    The delayed actions are stored in a queue and applied after N steps.
    """

    def __init__(self, env, delays=(3, 8, 1)):
        """
        delays: tuple of delay steps (in env timesteps) for each action dimension
                (LR delay, blood delay, norepi delay)
        """
        super().__init__(env)
        self.delays = np.array(delays, dtype=int)
        assert self.delays.shape[0] == 3, "Expecting 3 delays for LR, blood, norepi"
        # queues[i] holds the pending actions for that channel
        self.queues = [[] for _ in range(3)]
        self.default_action = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.queues = [[] for _ in range(3)]
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        delayed_action = np.zeros_like(action)

        # Push each action into its delay queue and pop the oldest one if full
        for i in range(3):
            self.queues[i].append(action[i])
            if len(self.queues[i]) > self.delays[i]:
                delayed_action[i] = self.queues[i].pop(0)
            else:
                # queue not yet filled → apply default (no infusion)
                delayed_action[i] = 0.0

        # Pass delayed actions to the inner environment
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        info["applied_action"] = delayed_action
        info["queued_action"] = action
        return obs, reward, terminated, truncated, info


class SmoothActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, alphas=(0.3, 0.15, 0.6)):
        """
        alphas: smoothing factors for each action dimension
                 0.0 = infinitely slow, 1.0 = instant response
        """
        super().__init__(env)
        self.alphas = np.array(alphas)
        self.prev_effective_action = np.zeros(self.env.action_space.shape)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_effective_action = np.zeros_like(self.prev_effective_action)
        return obs, info

    def step(self, action):
        # Apply smooth (gradual) transition toward new action
        effective_action = (
            (1 - self.alphas) * self.prev_effective_action
            + self.alphas * action
        )
        self.prev_effective_action = effective_action.copy()
        obs, reward, done, truncated, info = self.env.step(effective_action)
        return obs, reward, done, truncated, info
