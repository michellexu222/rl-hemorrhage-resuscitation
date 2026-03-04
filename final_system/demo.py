import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
## replace with appropriate paths
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python")
sys.path.insert(1, r"C:\Users\michellexu\builds\pulse-engine-conda\install\bin")
sys.path.insert(2, r"C:\Users\michellexu\builds\pulse-engine-conda\Innerbuild\src\python")
sys.path.insert(3, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(4, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(5, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")
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


def run_and_collect_live(base_env, eval_env, left_col, right_col, max_steps=100, deterministic=True,
                         seed=None, index=None):
    if index is not None:
        hem_sev = "high" if index < 25 else "low"
    else:
        hem_sev = None

    obs, reset_info = base_env.reset()
    sev = reset_info['sev']
    map_high = 110 if sev == "low" else 90

    # Gating
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
        severity = torch.argmax(output, dim=1)

    severity_str = 'Low' if severity.item() == 0 else 'High'
    agent_str = 'PPO' if severity == 0 else 'Recurrent PPO'

    with left_col:
        st.markdown(
            f"**Hem:** {reset_info['hem'][0]} &nbsp;|&nbsp; "
            f"**Sev:** {reset_info['hem'][1]:.3f} &nbsp;|&nbsp; "
            f"**Predicted:** {severity_str} &nbsp;|&nbsp; "
            f"**Agent:** {agent_str}",
            unsafe_allow_html=True
        )
        st.markdown("---")

    # Load expert model
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

    # Create placeholders in left column
    with left_col:
        status_placeholder = st.empty()
        st.markdown("**Vitals**")
        vitals_placeholder = st.empty()
        st.markdown("**Actions**")
        actions_placeholder = st.empty()
        st.markdown("**Metrics**")
        metrics_placeholder = st.empty()

    # Create placeholder for plot in right column
    with right_col:
        plot_placeholder = st.empty()

    while not done and step < max_steps:
        if recurrent:
            action, lstm_states = model.predict(obs_norm, state=lstm_states, deterministic=deterministic)
        else:
            action, _ = model.predict(obs_norm, deterministic=deterministic)

        a = np.array(action).flatten()

        obs, reward, terminated, truncated, info = base_env.step(action)

        if obs[1] > map_high:
            map_violations += 1

        obs_norm = eval_env.normalize_obs(obs)
        total_reward += reward
        done_flag = terminated or truncated

        obs_arr = np.array(obs).flatten()
        hr_val = float(obs_arr[0])
        map_val = float(obs_arr[1])
        sap_val = float(obs_arr[2])
        oxsat_val = float(obs_arr[3])
        etco2_val = float(obs_arr[4])
        resp_val = float(obs_arr[5])
        skin_temp_val = float(obs_arr[6])

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

        # left column updates
        with left_col:
            with status_placeholder.container():
                st.markdown(f"**Step:** {step} / {max_steps}")
                if done_flag:
                    outcome = info.get('o', 'unknown')
                    if outcome == 'stabilization':
                        st.success(f"**{outcome.upper()}**")
                    else:
                        st.error(f"**{outcome.upper()}**")

            with vitals_placeholder.container():
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    st.metric("MAP", f"{map_val:.0f}", delta=f"{map_val - 65:.0f}", delta_color="inverse")
                with r1c2:
                    st.metric("HR", f"{hr_val:.0f}")
                with r1c3:
                    st.metric("SpO₂", f"{oxsat_val:.1f}%")
                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    st.metric("SAP", f"{sap_val:.0f}")
                with r2c2:
                    st.metric("Shock Idx", f"{hr_val / sap_val:.2f}" if sap_val > 0 else "N/A")
                with r2c3:
                    st.metric("RespRate", f"{resp_val:.0f}")

            with actions_placeholder.container():
                a1, a2 = st.columns(2)
                with a1:
                    st.metric("Crystalloid", f"{cryst_rate_ml:.0f}")
                    st.metric("Blood", f"{blood_rate_ml:.0f}")
                with a2:
                    st.metric("Vasopressor", f"{vp_rate_ml:.3f}")
                    st.metric("Clot", f"{info.get('clot_frac', 0):.2f}")

            with metrics_placeholder.container():
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Reward", f"{total_reward:.1f}")
                with m2:
                    st.metric("MAP Viol.", map_violations)

        # Right column updates
        if step % 1 == 0 or done_flag:
            df = pd.DataFrame(records)
            fig = plot_episode_compact(df, info, total_reward, step, max_steps)
            with right_col:
                with plot_placeholder:
                    st.pyplot(fig)
            plt.close(fig)

        step += 1
        time.sleep(0.1)

        if done_flag:
            break

    return records, total_reward, map_violations, step, severity, reset_info, info


def plot_episode_compact(df, final_info, total_reward, current_step, max_steps):
    """
    4 dynamic plots (MAP, HR, Fluids, Clot)
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.15, top=0.93, bottom=0.08, left=0.14, right=0.97)

    ax0 = axes[0]
    ax0.plot(df["timestep"], df["MAP"], 'b-', linewidth=1.5)
    ## uncomment below for lines indicating MAP targets
    #ax0.axhline(y=65, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    #ax0.axhline(y=50, color='red', linestyle=':', alpha=0.4)
    #ax0.axhline(y=110, color='red', linestyle=':', alpha=0.4)
    ax0.set_ylabel("Mean Arterial Pressure (mmHg)", fontsize=8)
    ax0.set_title(f"Step {current_step}/{max_steps}  |  Reward: {total_reward:.1f}",
                  fontweight='bold', fontsize=9, pad=2)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(30, 120)
    ax0.tick_params(labelsize=7)

    if df["done"].any():
        done_idx = df[df["done"] == True].index[0]
        ax0.axvline(df.loc[done_idx, "timestep"], color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    ax1 = axes[1]
    ax1.plot(df["timestep"], df["HR"], 'r-', linewidth=1.5)
    ax1.set_ylabel("Heart Rate (BPM)", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 180)
    ax1.tick_params(labelsize=7)

    ax2 = axes[2]
    ax2.fill_between(df["timestep"], 0, df["cryst_ml_per_min"], step='post', alpha=0.7, label="Crystalloids", color='#87CEEB')
    ax2.fill_between(df["timestep"], 0, df["blood_ml_per_min"], step='post', alpha=0.7, label="Blood", color='#DC143C')
    ax2.set_ylabel("mL/min", fontsize=8)
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 450)
    ax2.tick_params(labelsize=7)

    ax3 = axes[3]
    ax3.plot(df["timestep"], df["clot_frac"], 'g-', linewidth=1.5)
    ax3.set_ylabel("Clot", fontsize=8)
    ax3.set_xlabel("Time (min)", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.tick_params(labelsize=7)

    return fig


### Streamlit

st.set_page_config(page_title="Hemorrhage RL Demo", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    h1 { margin-top: 0; margin-bottom: 0.3rem; font-size: 1.5rem; }
    h3 { margin-top: 0.3rem; margin-bottom: 0.1rem; }
    /* Tighten metric spacing */
    [data-testid="stMetric"] {
        padding: 0 !important;
        margin-bottom: -0.05rem !important;
    }
    [data-testid="stMetricLabel"] { font-size: 1.6rem !important; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
    [data-testid="stMetricDelta"] { font-size: 1.2rem !important; }
    /* Body text in left panel */
    .stMarkdown p { margin-bottom: 0.2rem; font-size: 1.15rem; }
    hr { margin: 0.3rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Hemorrhage Resuscitation RL Agent Demo")

# Run button & summary metrics shown at episode end
col_btn, col_reward, col_duration, col_outcome = st.columns([2, 1, 1, 1])
with col_btn:
    run_button = st.button('Run New Episode', type="primary", use_container_width=True)
summary_reward_ph   = col_reward.empty()
summary_duration_ph = col_duration.empty()
summary_outcome_ph  = col_outcome.empty()

st.markdown("---")

if run_button:
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("---")
        _c1, _c2 = st.columns([3, 3])
        with _c1:
            max_steps = st.slider("Max timesteps", 20, 100, 60)
        with _c2:
            st.write("")
            deterministic = st.checkbox("Deterministic actions", value=True)

    base_env = make_env()
    eval_env = DummyVecEnv([make_env])

    with st.spinner("Running episode"):
        records, total_reward, map_violations, final_step, severity, reset_info, final_info = \
            run_and_collect_live(base_env, eval_env, left_col=left_col,
                                 right_col=right_col, max_steps=max_steps,
                                 deterministic=deterministic, seed=None, index=None)

    # show summary metrics
    summary_reward_ph.metric("Final Reward", f"{total_reward:.2f}")
    summary_duration_ph.metric("Duration", f"{final_step} min")
    summary_outcome_ph.metric("Outcome", final_info.get('o', 'unknown'))

    df_final = pd.DataFrame(records)
    csv_data = df_final.to_csv(index=False)
    st.download_button(
        label="Download Episode Data (CSV)",
        data=csv_data,
        file_name=f"episode_{severity.item()}sev_{final_info.get('o', 'unk')}.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    max_steps = 60
    deterministic = True
    st.info("Click 'Run New Episode' to start the demo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Displays:**
        - Vital signs and agent actions (updated live)
        - Live plots of physiological state 
        - Outcome (Stabilization or death) and summary metrics shown at end
        """)
    with col2:
        st.markdown("""
        **What happens:**
        1. Gating network predicts hemorrhage severity
        2. Patient is routed to either the high or low severity expert
        3. Agent sets crystalloid, blood, vasopressor rates each minute
        4. Environment simulates patient response; shown vital signs are updated
        """)