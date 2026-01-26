import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(1, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(2, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import numpy as np
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import load

# save as eval_plot.py and run in same project / venv where Pulse and model exist

import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from env.env_wrappers import SmoothActionDelayWrapper

from env.hemorrhage_env import HemorrhageEnv
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
# # load saved normalization so model sees same scaling as during training
# eval_env = VecNormalize.load(venv_stats_path, eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

#model = PPO.load(model_path, env=eval_env)
#model = PPO.load(model_path, env=eval_env)
#model = RecurrentPPO.load(model_path)

# ---------- Run one deterministic episode and collect history ----------
def run_and_collect(base_env, eval_env, max_steps=100, deterministic=True, seed=None, index=None):
    #obs, info = base_env.reset(seed=seed)
    if index is not None:
        hem_sev = "high" if index < 25 else "low"
    else: hem_sev = None
    obs, reset_info = base_env.reset(seed=seed, sev=hem_sev)
    #obs[1] = 80 # changed for feature ablation - no map
    #obs[0] = 80 # changed for feature ablation - no hr
    #obs[2] = 105 # changed for feature ablation - no sap
    #bs[3] = 98 # changed for feature ablation - no oxsat
    #obs[4] = 40 # changed for feature ablation - no etco2
    #obs[5] = 16 # changed for feature ablation - no resp rate
    obs[6] = 36.5 # changed for feature ablation - no skin temp
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
        if obs[1] > map_high: map_violations += 1
        #obs[1] = 80 # changed for feature ablation - no map
        #obs[0] = 80 # changed for feature ablation - no hr
        #obs[2] = 105 # changed for feature ablation - no sap
        #obs[3] = 98 # changed for feature ablation - no oxsat
        #obs[4] = 40 # changed for feature ablation - no etco2
        #obs[5] = 16 # changed for feature ablation - no resp rate
        obs[6] = 36.5 # changed for feature ablation - no skin temp

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

# ---------- Plotting ----------
def plot_episode(df, save_fig=None, show=True):
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.5, 1, 1, 1.5, 1], hspace=0.3)

    # Panel 1: MAP vs time
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(df["timestep"], df["MAP"], marker='o', linestyle='-', label="MAP", linewidth=2)
    ax0.set_ylabel("MAP (mmHg)")
    ax0.set_xlabel("Timestep (min)")
    ax0.set_title(f"Episode MAP and actions — total_reward={total_reward:.2f} outcome={final_info.get('o', 'N/A')}")
    ax0.grid(True)
    # mark termination timestep
    done_idx = df.index[df["done"]==True].tolist()
    if done_idx:
        idx = done_idx[0]
        ax0.axvline(df.loc[idx, "timestep"], color='red', linestyle='--', label='terminal')
        ax0.scatter(df.loc[idx, "timestep"], df.loc[idx, "MAP"], color='red', zorder=5)

    # Panel 2: Fluid infusion rates (crystalloid & blood)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.step(df["timestep"], df["cryst_ml_per_min"], where='post', label="Crystalloid (mL/min)")
    ax1.step(df["timestep"], df["blood_ml_per_min"], where='post', label="Blood (mL/min)")
    ax1.set_ylabel("Volume rate (mL/min)")
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Panel 3: Vasopressor
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.step(df["timestep"], df["vp_ml_per_min"], where='post', label="Vasopressor rate (mL/min)")
    ax2.set_ylabel("Vasopressor rate")
    ax2.set_xlabel("Timestep (min)")
    ax2.grid(True)

    # Panel 4: Heart Rate
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.step(df["timestep"], df["HR"], where='post', label="Heart Rate")
    ax3.set_ylabel("Heart Rate")
    ax3.set_xlabel("Timestep (min)")
    ax3.grid(True)

    # Panel 5: Shock Index
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax4.step(df["timestep"], df["clot_frac"], where='post', label="Clot Fraction")
    ax4.set_ylabel("Clot Fraction")
    ax4.set_xlabel("Timestep (min)")
    ax4.grid(True)

    # overlay small text labels for actions
    if len(df) <= 60:
        for _, row in df.iterrows():
            t = row["timestep"]
            ax0.text(t, row["MAP"] + 2, f"a={row['action_0_norm']:.2f},{row['action_1_norm']:.2f},{row['action_2_norm']:.2f}",
                     fontsize=6, rotation=45)

    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")

    if show:
        plt.show()
    plt.close(fig)

# # ---------- Run & plot one episode ----------
# records, total_reward, map_violations, length, reset_info, final_info = run_and_collect(model, base_env, eval_env, max_steps=400, deterministic=True)
#
# # save CSV
# df = pd.DataFrame(records)
# out_csv = os.path.join(parent_dir, "eval_episode.csv")
# df.to_csv(out_csv, index=False)
# print(f"Saved episode CSV to: {out_csv}")
#
# plot_path = os.path.join(parent_dir, "episode_map_actions_mod_sev_bloodcost.png")
# plot_episode(df, save_fig=plot_path, show=True)

eval_data_path = os.path.join(script_dir, "feature_ablation", "MoE", "csvs", "no_skintemp.csv") # changed for feature ablation
if not os.path.exists(eval_data_path):
    with open(eval_data_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "organ", "severity", "gating", "patient", "outcome", "length", "blood_total", "cryst_total", "vp_total", "map_violations"])

# ---------- Run & calculate metrics for multiple episodes ----------
# vars for stabilization metrics
num_eps = 50
base_seed = 42
n_death = 0
n_stable_liver = 0
n_stable_spleen = 0
total_liver = 0
total_spleen = 0

# vars for reward metric
total_return = 0

# vars for resource metrics
cryst_used = []
blood_used = []
vp_used = []
fluids_used = []
stabilized_fluids_used = []  # fluids used in stabilized episodes

for i in range(num_eps):
    seed = base_seed + i
    records, total_reward, map_violations, length, gating, reset_info, final_info = run_and_collect(base_env, eval_env, max_steps=400, deterministic=True, seed=seed, index=i)

    # -------- save CSV and plot episode --------
    df = pd.DataFrame(records)
    #out_csv = os.path.join(script_dir, "feature_ablation", "MoE", "csvs", "no_map.csv") # changed for feature ablation
    #df.to_csv(out_csv, index=False)
    #print(f"Saved episode CSV to: {out_csv}")

    plot_path = os.path.join(parent_dir, "final_system", "feature_ablation", "MoE", "plots", "no_skintemp") # changed for feature ablation
    os.makedirs(plot_path, exist_ok=True)
    plot_path = os.path.join(parent_dir, "final_system", "feature_ablation", "MoE", "plots", "no_skintemp", f"eval_episode{i+1}.png")
    plot_episode(df, save_fig=plot_path, show=False)

    # "id", "organ", "severity", "gating", "patient", "outcome", "length", "blood_total", "cryst_total", "vp_total", "map_violations"
    with open(eval_data_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            i,
            reset_info['hem'][0],
            reset_info['hem'][1],
            gating,
            reset_info['state_file'],
            final_info['o'],
            length,
            sum(r["blood_ml_per_min"] for r in records),
            sum(r["cryst_ml_per_min"] for r in records),
            sum(r["vp_ml_per_min"] for r in records),
            map_violations
        ])

    # increment outcome counts
    if final_info['o'] in ['death', 'failed to advance', 'death (HR)']:
        n_death += 1

        if final_info['hem'][0] == 'liver': total_liver += 1
        else: total_spleen += 1

    elif final_info['o'] == 'stabilization':
        if final_info['hem'][0] == 'spleen':
            n_stable_spleen += 1
            total_spleen += 1
        elif final_info['hem'][0] == 'liver':
            n_stable_liver += 1
            total_liver += 1

    elif final_info['o'] == 'truncated': pass # truncated episodes ignored
    else: print("Unknown outcome:", final_info['o'])

    # accumulate return
    total_return += total_reward

    # track resource usage
    total_cryst = sum(r["cryst_ml_per_min"] for r in records) # total crystalloid infused in episode
    total_blood = sum(r["blood_ml_per_min"] for r in records) # total blood infused in episode
    total_vp = sum(r["vp_ml_per_min"] for r in records)         # total vasopressor infused in episode
    total_fluids = total_cryst + total_blood

    cryst_used.append(total_cryst)
    blood_used.append(total_blood)
    vp_used.append(total_vp)
    fluids_used.append(total_fluids)

    if final_info['o'] == "stabilization": stabilized_fluids_used.append(total_fluids)
    print(f"Episode {i+1} done. Outcome: {final_info['o']}")

# ---------- Print stabilization metrics ----------
n_stable = n_stable_liver + n_stable_spleen
total = total_liver + total_spleen
print(f"% stabilized total: {n_stable}/{total} = {n_stable/total*100:.1f}%")
print(f"% stabilized liver: {n_stable_liver}/{total_liver} = {n_stable_liver/total_liver*100:.1f}%")
print(f"% stabilized spleen: {n_stable_spleen}/{total_spleen} = {n_stable_spleen/total_spleen*100:.1f}%")
print(f"% died: {n_death}/{total} = {n_death/total*100:.1f}%")

# ---------- Print reward metric ----------
average_return = total_return / num_eps
print(f"Average return over {num_eps} episodes: {average_return:.2f}")

# ---------- Print resource usage metrics ----------
print(f"Median crystalloid used: {np.median(cryst_used):.1f} mL (mean {np.mean(cryst_used):.1f} mL)")
print(f"Median blood used: {np.median(blood_used):.1f} mL (mean {np.mean(blood_used):.1f} mL)")
print(f"Median vasopressor used: {np.median(vp_used):.3f} mL (mean {np.mean(vp_used):.3f} mL)")

print(f"Resource efficiency: {np.mean(stabilized_fluids_used):.2f} mL")
