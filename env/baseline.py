from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from hemorrhage_env import HemorrhageEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # SB3 automatically aggregates Monitor stats in 'infos'
        for info in self.locals["infos"]:
            if "episode" in info.keys():
                print(f"Episode Reward: {info['episode']['r']} | "
                      f"Episode Length: {info['episode']['l']} | "
                      f"Episode Outcome: {info['o']} | "
                      f"Hemorrhage: {info['hem']}")
        return True


torch.manual_seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "..", "configs")
os.makedirs(target_dir, exist_ok=True)

def make_env():
    env = HemorrhageEnv()
    env = Monitor(env)
    return env

venv = DummyVecEnv([make_env])
venv.seed(42)
# Add normalization (running mean/std for obs)
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)
model = PPO("MlpPolicy", venv, seed=42, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=100, callback=RewardCallback())

venv.save(os.path.join(target_dir, "venv_stats.pkl"))

