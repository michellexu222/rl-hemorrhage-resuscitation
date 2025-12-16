import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(1, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(2, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")

from hemorrhage_env import HemorrhageEnv
import torch
import numpy as np
import os
from gating.model import GatingNet
from sklearn.preprocessing import StandardScaler
from joblib import load
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def make_env():
    env = HemorrhageEnv(eval=True)
    # env = SmoothActionDelayWrapper(env)
    env = Monitor(env)
    return env

def run_episode(model, base_env, eval_env, reset_obs):
    done = False
    step = 0
    total_reward = 0.0
    obs_norm = eval_env.normalize_obs(reset_obs)
    while not done:
        action, _ = model.predict(obs_norm, deterministic=True)
        a = np.array(action).flatten()

        obs, reward, terminated, truncated, info = base_env.step(action)
        obs_norm = eval_env.normalize_obs(obs)
        total_reward += reward
        done = terminated or truncated

    return info['o'], total_reward

# ---- environment setup ----
base_env = HemorrhageEnv()
eval_env = DummyVecEnv([make_env])

# ---- gating ----
obs, info = base_env.reset()
print(f"hemorrhage: {info['hem']}")

gating_obs = torch.tensor((info["bv1"] - info["bv2"], info["bv2"] - info["bv3"]))
gating_obs = gating_obs.unsqueeze(0)
print(gating_obs)
scaler = load(r'C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_scaler.pkl')
gating_obs = scaler.transform(gating_obs)
gating_obs = torch.tensor(gating_obs, dtype=torch.float32)
print(gating_obs)

gating_model = GatingNet()
gating_model.load_state_dict(torch.load(r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_model.pth", weights_only=True))
gating_model.eval()
with torch.no_grad():
    output = gating_model(gating_obs)
    output = torch.softmax(output, dim=-1)
    print(output)
    severity = torch.argmax(output, dim=1) # 0 = low, 1 = high
    print(severity)

if severity == 0:
    venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_ppo_modsev_4.pkl")
    model_path = os.path.join(parent_dir, "models", "ppo_modsev_4.zip")

else:
    venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_rppo_highsev_1.pkl")
    model_path = os.path.join(parent_dir, "models", "rppo_highsev_1.zip")

# load saved normalization so model sees same scaling as during training
eval_env = VecNormalize.load(venv_stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False

model = PPO.load(model_path, env=eval_env)

outcome, reward = run_episode(model, base_env, eval_env, obs)
print(f"Severity = {severity}, Outcome: {outcome}, Total Reward: {reward}")





