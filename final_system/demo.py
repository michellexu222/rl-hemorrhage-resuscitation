import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python")
sys.path.insert(1, r"C:\Users\michellexu\builds\pulse-engine-conda\install\bin")
sys.path.insert(2, r"C:\Users\michellexu\builds\pulse-engine-conda\Innerbuild\src\python")
sys.path.insert(3, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(4, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(5, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import streamlit as st
import torch
import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import load

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from env.hemorrhage_env import HemorrhageEnv
from gating.model import GatingNet


def make_env():
    env = HemorrhageEnv(eval=True)
    env = Monitor(env)
    return env

def run_and_collect_live(base_env, eval_env, max_steps=100, deterministic=True, seed=None, index=None):
    """
    Run episode with live Streamlit updates.
    Yields data after each step for real-time display.
    """
    if index is not None:
        hem_sev = "high" if index < 25 else "low"
    else:
        hem_sev = None

    obs, reset_info = base_env.reset()
    sev = reset_info['sev']
    map_high = 110 if sev == "low" else 90

    # Display initial info
    st.write(f"**Episode Started**")
    st.write(f"- Hemorrhage: {reset_info['hem'][0]} (severity {reset_info['hem'][1]:.3f})")
    st.write(f"- Patient: {reset_info['state_file']}")

    # ---- Gating ----
    gating_obs = torch.tensor((reset_info["bv1"] - reset_info["bv2"],
                               reset_info["bv2"] - reset_info["bv3"]))
    gating_obs = gating_obs.unsqueeze(0)

    scaler = load(
        r'C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_scaler1.pkl')
    gating_obs = scaler.transform(gating_obs)
    gating_obs = torch.tensor(gating_obs, dtype=torch.float32)

    gating_model = GatingNet()
    gating_model.load_state_dict(torch.load(
        r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating\gating_model1.pth",
        weights_only=True))
    gating_model.eval()

    with torch.no_grad():
        output = gating_model(gating_obs)
        output = torch.softmax(output, dim=-1)
        severity = torch.argmax(output, dim=1)  # 0 = low, 1 = high

    severity_str = 'low' if severity.item() == 0 else 'high'
    st.write(f"**Gating prediction: {severity_str}**")
    st.write("---")

    # Load appropriate model
    if severity == 0:
        venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_ppo_modsev_4.pkl")
        model_path = os.path.join(parent_dir, "models", "ppo_modsev_4.zip")
        eval_env = VecNormalize.load(venv_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        model = PPO.load(model_path, env=eval_env)
        recurrent = False
    else:
        venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_rppo_highsev_2.pkl")
        model_path = os.path.join(parent_dir, "models", "rppo_highsev_2.zip")
        eval_env = VecNormalize.load(venv_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        model = RecurrentPPO.load(model_path)
        recurrent = True

    obs_norm = eval_env.normalize_obs(obs)
    lstm_states = None
    done = False
    step = 0
    records = []
    total_reward = 0.0
    map_violations = 0

    # Create placeholders for live updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    plot_placeholder = st.empty()

    while not done and step < max_steps:
        # Get action
        if recurrent:
            action, lstm_states = model.predict(obs_norm, state=lstm_states, deterministic=deterministic)
        else:
            action, _ = model.predict(obs_norm, deterministic=deterministic)

        a = np.array(action).flatten()

        # Step environment
        obs, reward, terminated, truncated, info = base_env.step(action)

        if obs[1] > map_high:
            map_violations += 1

        # Feature ablation (uncomment as needed)
        # obs[6] = 33  # skin temp ablation

        obs_norm = eval_env.normalize_obs(obs)
        total_reward += reward
        done_flag = terminated or truncated

        # Extract observations
        obs_arr = np.array(obs).flatten()
        hr_val = float(obs_arr[0])
        map_val = float(obs_arr[1])
        sap_val = float(obs_arr[2])
        oxsat_val = float(obs_arr[3])
        etco2_val = float(obs_arr[4])
        resp_val = float(obs_arr[5])
        skin_temp_val = float(obs_arr[6])

        # Convert actions
        cryst_rate_ml = a[0] * 400
        blood_rate_ml = a[1] * 400
        vp_rate_ml = a[2] * 0.04

        # Record data
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
            "ShockIndex": hr_val / sap_val if sap_val > 0 else 0,
            "HR": hr_val,
            "OxSat": oxsat_val,
            "EtCO2": etco2_val,
            "RespRate": resp_val,
            "SkinTemp": skin_temp_val,
            "reward": float(reward),
            "done": done_flag,
            "clot_frac": info.get("clot_frac", 0),
            "sev_new": info.get("sev_new", 0),
        }
        records.append(rec)

        # ===== LIVE UPDATE DISPLAY =====

        # Update status text
        with status_placeholder.container():
            st.write(f"### Timestep {step} / {max_steps}")
            if done_flag:
                outcome = info.get('o', 'unknown')
                st.success(f"**Episode ended: {outcome}**")

        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAP", f"{map_val:.1f} mmHg",
                          delta=f"{map_val - 80:.1f}" if step > 0 else None)
                st.metric("HR", f"{hr_val:.0f} bpm")
            with col2:
                st.metric("SAP", f"{sap_val:.1f} mmHg")
                st.metric("Shock Index", f"{hr_val / sap_val:.2f}" if sap_val > 0 else "N/A")
            with col3:
                st.metric("Crystalloid", f"{cryst_rate_ml:.0f} mL/min")
                st.metric("Blood", f"{blood_rate_ml:.0f} mL/min")
            with col4:
                st.metric("Vasopressor", f"{vp_rate_ml:.3f} mL/min")
                st.metric("Reward", f"{reward:.2f}")

        # Update plot (every N steps to avoid slowdown)
        if step % 1 == 0 or done_flag:  # Update every step (or every 2 steps if too slow)
            df = pd.DataFrame(records)
            fig = plot_episode_live(df, info, total_reward)
            with plot_placeholder:
                st.pyplot(fig)
            plt.close(fig)

        step += 1

        # Small delay for visibility (optional, remove for max speed)
        time.sleep(0.1)

        if done_flag:
            break

    return records, total_reward, map_violations, step, severity, reset_info, info


def plot_episode_live(df, final_info, total_reward):
    """
    Create a compact live-updating plot
    Returns the figure object for Streamlit to display
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Panel 0: MAP + HR
    ax0 = axes[0]
    ax0.plot(df["timestep"], df["MAP"], 'b-', linewidth=2, label="MAP")
    ax0.axhline(y=65, color='orange', linestyle='--', alpha=0.5, label="Target (65)")
    ax0.set_ylabel("MAP (mmHg)")
    ax0.set_title(f"Episode Progress — Reward: {total_reward:.2f}")
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Mark termination
    if df["done"].any():
        done_idx = df[df["done"] == True].index[0]
        ax0.axvline(df.loc[done_idx, "timestep"], color='red', linestyle='--', alpha=0.7)

    # Panel 1: Heart Rate
    ax1 = axes[1]
    ax1.plot(df["timestep"], df["HR"], 'r-', linewidth=2, label="HR")
    ax1.set_ylabel("HR (bpm)")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Fluid Actions
    ax2 = axes[2]
    ax2.fill_between(df["timestep"], 0, df["cryst_ml_per_min"],
                     step='post', alpha=0.6, label="Crystalloid", color='lightblue')
    ax2.fill_between(df["timestep"], 0, df["blood_ml_per_min"],
                     step='post', alpha=0.6, label="Blood", color='darkred')
    ax2.set_ylabel("Fluid Rate (mL/min)")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Clot Fraction
    ax3 = axes[3]
    ax3.plot(df["timestep"], df["clot_frac"], 'g-', linewidth=2, label="Clot Fraction")
    ax3.set_ylabel("Clot Fraction")
    ax3.set_xlabel("Time (minutes)")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Hemorrhage RL Demo", layout="wide")
st.title("Hemorrhage Resuscitation RL Agent - Live Demo")

st.markdown("""
This demo shows a reinforcement learning agent managing hemorrhagic shock in real-time.
The agent chooses how much crystalloid, blood, and vasopressor to administer each minute.
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    max_steps = st.slider("Max timesteps", 20, 100, 60)
    deterministic = st.checkbox("Deterministic actions", value=True)
    show_delay = st.checkbox("Add display delay (slower but easier to watch)", value=True)

    st.markdown("---")
    st.markdown("**Model Info:**")
    st.markdown("- Low severity: PPO")
    st.markdown("- High severity: Recurrent PPO")

# Main content
if st.button('Run New Episode', type="primary"):
    base_env = make_env()
    eval_env = DummyVecEnv([make_env])

    with st.spinner("Running episode..."):
        records, total_reward, map_violations, final_step, severity, reset_info, final_info = \
            run_and_collect_live(base_env, eval_env, max_steps=max_steps,
                                 deterministic=deterministic, seed=None, index=None)

    # Final summary
    st.success("Episode complete!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reward", f"{total_reward:.2f}")
    with col2:
        st.metric("Duration", f"{final_step} min")
    with col3:
        st.metric("Outcome", final_info.get('o', 'unknown'))
    with col4:
        st.metric("MAP Violations", map_violations)

    # Download data option
    df_final = pd.DataFrame(records)
    csv = df_final.to_csv(index=False)
    st.download_button(
        label="Download Episode Data (CSV)",
        data=csv,
        file_name=f"episode_{severity.item()}sev.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Click 'Run New Episode' to start!")
    st.markdown("""
    ### What you'll see:
    - **Real-time vital signs**: MAP, HR, SAP, Shock Index
    - **Agent actions**: Crystalloid, blood, and vasopressor rates
    - **Live plots**: Updated every timestep
    - **Episode outcome**: Stabilization or death
    """)