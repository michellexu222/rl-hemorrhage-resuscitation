from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from ich_env import IntracranialHemorrhageEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
os.makedirs(target_dir, exist_ok=True)  # make sure it exists

env = IntracranialHemorrhageEnv(state_file=os.path.join(target_dir, f"Patient0@0s.json"))

venv = DummyVecEnv([lambda: IntracranialHemorrhageEnv("patient.json")]) # labmda: ... is a function with no args, when called returns a fresh ICHEnv

# Add normalization (running mean/std for obs, optional for rewards)
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.)

# model = PPO("MlpPolicy", venv, verbose=1)
# model.learn(total_timesteps=100000)
#
# # Save running statistics so evaluation uses the same normalization
# venv.save("venv_stats.pkl")
# model.save("ppo_hemorrhage")
check_env(env, warn=True)
