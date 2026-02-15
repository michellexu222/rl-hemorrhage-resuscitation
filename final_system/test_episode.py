import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np
import csv
import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from env.env_wrappers import SmoothActionDelayWrapper
from env.hemorrhage_env import HemorrhageEnv
from sb3_contrib import RecurrentPPO
import torch

# save as eval_plot.py and run in same project / venv where Pulse and model exist
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.env_wrappers import SmoothActionDelayWrapper
from pid_baseline import PIDBaseline
from gating.model import GatingNet

def make_env():
    env = HemorrhageEnv(eval=True)
    # env = SmoothActionDelayWrapper(env)
    env = Monitor(env)
    return env

# ---------- Load eval env & model ----------
# venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_ppo_modsev_4.pkl")
# model_path = os.path.join(parent_dir, "models", "ppo_modsev_4.zip")

# venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_rppo_highsev_2.pkl")
# model_path = os.path.join(parent_dir, "models", "rppo_highsev_2.zip")

recurrent = False

base_env = make_env()
eval_env = DummyVecEnv([make_env])

# ---------- Run one deterministic episode and collect history ----------
def run_and_collect(base_env, eval_env, max_steps=100, deterministic=True, seed=None, index=None):
    #obs, info = base_env.reset(seed=seed)
    if index is not None:
        hem_sev = "high" if index < 25 else "low"
    else: hem_sev = None
    obs, reset_info = base_env.reset(seed=seed, sev=hem_sev)
    print(obs)
    #obs[1] = 80 # changed for feature ablation - no map
    #obs[0] = 80 # changed for feature ablation - no hr
    #obs[2] = 105 # changed for feature ablation - no sap
    #bs[3] = 98 # changed for feature ablation - no oxsat
    #obs[4] = 40 # changed for feature ablation - no etco2
    #obs[5] = 16 # changed for feature ablation - no resp rate
    #obs[6] = 36.5 # changed for feature ablation - no skin temp
    sev = reset_info['sev']
    map_high = 110 if sev == "low" else 90

    print(f"hemorrhage: {reset_info['hem']}, patient: {reset_info['state_file']}")
    #obs_norm = eval_env.normalize_obs(obs)

    # ---- gating ----
    gating_obs = torch.tensor((reset_info["bv1"] - reset_info["bv2"], reset_info["bv2"] - reset_info["bv3"]))
    gating_obs = gating_obs.unsqueeze(0)
    #print(gating_obs)
    scaler = load(
        r'C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_scaler1.pkl')
    gating_obs = scaler.transform(gating_obs)
    gating_obs = torch.tensor(gating_obs, dtype=torch.float32)
    #print(gating_obs)

    gating_model = GatingNet()
    gating_model.load_state_dict(torch.load(
        r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_model1.pth",
        weights_only=True))
    gating_model.eval()
    with torch.no_grad():
        output = gating_model(gating_obs)
        output = torch.softmax(output, dim=-1)
        #print(output)
        severity = torch.argmax(output, dim=1)  # 0 = low, 1 = high
        print(severity)

    if severity == 0:
        venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_ppo_modsev_4.pkl")
        model_path = os.path.join(parent_dir, "models", "ppo_modsev_4.zip")

        # load saved normalization so model sees same scaling as during training
        eval_env = VecNormalize.load(venv_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        model = PPO.load(model_path, env=eval_env)
        recurrent = False
    else:
        venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_rppo_highsev_2.pkl")
        model_path = os.path.join(parent_dir, "models", "rppo_highsev_2.zip")

        # load saved normalization so model sees same scaling as during training
        eval_env = VecNormalize.load(venv_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        model = RecurrentPPO.load(model_path)
        recurrent = True

    obs_norm = eval_env.normalize_obs(obs)

    lstm_states = None # RNN state initially
    done = False
    step = 0
    records = []
    total_reward = 0.0
    map_violations = 0

    while not done and step < max_steps:
        if recurrent:
            action, lstm_states = model.predict(obs_norm, state=lstm_states, deterministic=deterministic)
        else:
            action, _ = model.predict(obs_norm, deterministic=deterministic)
        a = np.array(action).flatten()

        obs, reward, terminated, truncated, info = base_env.step(action)
        print(obs)
        if obs[1] > map_high: map_violations += 1
        #obs[1] = 80 # changed for feature ablation - no map
        #obs[0] = 80 # changed for feature ablation - no hr
        #obs[2] = 105 # changed for feature ablation - no sap
        #obs[3] = 98 # changed for feature ablation - no oxsat
        #obs[4] = 40 # changed for feature ablation - no etco2
        #obs[5] = 16 # changed for feature ablation - no resp rate
        #obs[6] = 36.5 # changed for feature ablation - no skin temp

        obs_norm = eval_env.normalize_obs(obs)
        total_reward += reward
        done_flag = terminated or truncated

        # # Try to get the underlying state (MAP etc.) from the env:
        # # Option A: if your wrapped env stores prev_obs on the inner env object:
        # try:
        #     inner = env.envs[0].env  # DummyVecEnv -> Monitor -> HemorrhageEnv
        #     state_dict = getattr(inner, "prev_obs", None)
        #     print(state_dict)
        # except Exception:
        #     state_dict = None

        # try to read MAP from observation array if present
        map_val = None
        hr_val = None
        sap_val = None
        bv_val = None
        state_dict=None
        if state_dict:
            map_val = state_dict.get("MeanArterialPressure", None)
            hr_val = state_dict.get("HeartRate", None)
            sap_val = state_dict.get("SystolicArterialPressure", None)
            bv_val = state_dict.get("BloodVolume", None)
        else:

            try:
                obs_arr = np.array(obs).flatten()
                # obs: [HR, MAP, SAP, OxSat, EndTidalCO2Pressure, Resp rate, skin temp, age, bmi, sex]
                hr_val = float(obs_arr[0])
                map_val = float(obs_arr[1])
                sap_val = float(obs_arr[2])

            except Exception:
                pass

        # Convert normalized actions back
        cryst_rate_ml = a[0] * 400
        blood_rate_ml = a[1] * 400
        vp_rate_ml = a[2] * 0.04

        rec = {
            "timestep": step,
            "action_0_norm": float(a[0]),
            "action_1_norm": float(a[1]),
            "action_2_norm": float(a[2]),
            "cryst_ml_per_min": cryst_rate_ml,
            "blood_ml_per_min": blood_rate_ml,
            "vp_ml_per_min": vp_rate_ml,
            "MAP": map_val,
            "SAP": sap_val,
            "ShockIndex": hr_val / sap_val,
            "HR": hr_val,
            "BloodVolume": bv_val,
            "reward": float(reward),
            "done": done_flag,
            "clot_frac": info["clot_frac"],
            "sev_new": info["sev_new"],
            "info": info if isinstance(info, dict) else {}
        }
        records.append(rec)
        step += 1
        if done_flag:
            break

    return records, total_reward, map_violations, step, severity, reset_info, info

run_and_collect(base_env, eval_env)