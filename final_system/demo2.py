import sys
import os
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd
import time

# Add paths to system
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python")
sys.path.insert(1, r"C:\Users\michellexu\builds\pulse-engine-conda\install\bin")
sys.path.insert(2, r"C:\Users\michellexu\builds\pulse-engine-conda\Innerbuild\src\python")
sys.path.insert(3, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(4, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(5, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.hemorrhage_env import HemorrhageEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gating.model import GatingNet


def make_env():
    """Create and wrap the hemorrhage environment."""
    env = HemorrhageEnv(eval=True)
    env = Monitor(env)
    return env


def run_episode_with_live_updates(base_env, eval_env, max_steps=100, deterministic=True):
    """
    Enhanced version with progress tracking and structured updates.
    Includes full gating logic and model loading.
    """
    # Initialize episode
    obs, reset_info = base_env.reset()
    sev = reset_info['sev']
    map_high = 110 if sev == "low" else 90

    # Setup UI containers
    st.markdown("---")

    # Header row
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    with header_col1:
        st.subheader("🏥 Episode Progress")
    with header_col2:
        severity_badge = st.empty()
    with header_col3:
        progress_text = st.empty()

    # Display initial episode info
    st.write(f"**Hemorrhage:** {reset_info['hem'][0]} (severity {reset_info['hem'][1]:.3f})")
    st.write(f"**Patient:** {reset_info['state_file']}")

    # Progress bar
    progress_bar = st.progress(0)

    # Create tabs for organized display
    tab1, tab2, tab3 = st.tabs(["📊 Live Vitals", "💉 Actions", "📈 Full Timeline"])

    with tab1:
        vitals_container = st.container()

    with tab2:
        actions_container = st.container()

    with tab3:
        plot_placeholder = st.empty()

    # Status message area
    status_placeholder = st.empty()

    # ===== GATING MODEL LOGIC =====
    with st.spinner("Running gating model..."):
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

    severity_str = 'HIGH' if severity.item() == 1 else 'LOW'
    st.write(f"**Gating Model Prediction:** {severity_str} severity")
    st.write("---")

    # Update severity badge
    severity_badge.markdown(
        f"<div style='background-color: {'#ff4444' if severity.item() == 1 else '#44ff44'}; "
        f"padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>"
        f"SEVERITY: {severity_str}</div>",
        unsafe_allow_html=True
    )

    # ===== LOAD APPROPRIATE RL MODEL =====
    with st.spinner(f"Loading {severity_str} severity expert..."):
        if severity.item() == 0:  # Low severity
            venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_ppo_modsev_4.pkl")
            model_path = os.path.join(parent_dir, "models", "ppo_modsev_4.zip")

            # Load saved normalization so model sees same scaling as during training
            eval_env = VecNormalize.load(venv_stats_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

            model = PPO.load(model_path, env=eval_env)
            recurrent = False
            st.success("✅ Loaded Low-Severity Expert (PPO)")
        else:  # High severity
            venv_stats_path = os.path.join(parent_dir, "venv_stats", "venv_stats_rppo_highsev_2.pkl")
            model_path = os.path.join(parent_dir, "models", "rppo_highsev_2.zip")

            # Load saved normalization so model sees same scaling as during training
            eval_env = VecNormalize.load(venv_stats_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

            model = RecurrentPPO.load(model_path)
            recurrent = True
            st.success("✅ Loaded High-Severity Expert (Recurrent PPO)")

    # Initialize tracking
    obs_norm = eval_env.normalize_obs(obs)
    lstm_states = None
    records = []
    total_reward = 0.0
    map_violations = 0
    step = 0
    done = False

    # Main episode loop
    while not done and step < max_steps:
        # Update progress
        progress = (step + 1) / max_steps
        progress_bar.progress(progress)
        progress_text.markdown(f"**Step {step + 1}/{max_steps}**")

        # Get action
        if recurrent:
            action, lstm_states = model.predict(obs_norm, state=lstm_states,
                                                deterministic=deterministic)
        else:
            action, _ = model.predict(obs_norm, deterministic=deterministic)

        a = np.array(action).flatten()

        # Step environment
        obs, reward, terminated, truncated, info = base_env.step(action)

        # Check MAP violations
        if obs[1] > map_high:
            map_violations += 1

        # Feature ablation (uncomment as needed for testing)
        # obs[0] = 80    # HR ablation
        # obs[1] = 80    # MAP ablation
        # obs[2] = 105   # SAP ablation
        # obs[3] = 0.97  # OxSat ablation
        # obs[4] = 33.5  # EtCO2 ablation
        # obs[5] = 12    # RespRate ablation
        # obs[6] = 33    # SkinTemp ablation

        obs_norm = eval_env.normalize_obs(obs)
        total_reward += reward
        done = terminated or truncated

        # Extract state
        obs_arr = np.array(obs).flatten()
        hr = float(obs_arr[0])
        map_val = float(obs_arr[1])
        sap = float(obs_arr[2])
        oxsat = float(obs_arr[3])
        etco2 = float(obs_arr[4])
        resp_rate = float(obs_arr[5])
        skin_temp = float(obs_arr[6])

        # Convert actions
        cryst = a[0] * 400
        blood = a[1] * 400
        vp = a[2] * 0.04

        # Store record
        rec = {
            "timestep": step,
            "action_0_norm": float(a[0]),
            "action_1_norm": float(a[1]),
            "action_2_norm": float(a[2]),
            "HR": hr,
            "MAP": map_val,
            "SAP": sap,
            "OxSat": oxsat,
            "EtCO2": etco2,
            "RespRate": resp_rate,
            "SkinTemp": skin_temp,
            "ShockIndex": hr / sap if sap > 0 else 0,
            "cryst_ml_per_min": cryst,
            "blood_ml_per_min": blood,
            "vp_ml_per_min": vp,
            "reward": float(reward),
            "done": done,
            "clot_frac": info.get("clot_frac", 0),
            "sev_new": info.get("sev_new", 0)
        }
        records.append(rec)

        # ===== LIVE UPDATES =====

        # Update vitals display (Tab 1)
        with vitals_container:
            # Create gauge-style metrics
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)

            with vcol1:
                map_delta = map_val - 65
                map_color = "normal" if 55 <= map_val <= 75 else "inverse"
                st.metric("MAP", f"{map_val:.1f}", f"{map_delta:+.1f}", delta_color=map_color)
                st.metric("SAP", f"{sap:.1f}")

            with vcol2:
                hr_color = "normal" if hr < 120 else "inverse"
                st.metric("Heart Rate", f"{hr:.0f}", delta_color=hr_color)
                si = hr / sap if sap > 0 else 0
                si_color = "normal" if si < 1.0 else "inverse"
                st.metric("Shock Index", f"{si:.2f}", delta_color=si_color)

            with vcol3:
                st.metric("O₂ Sat", f"{oxsat * 100:.1f}%")
                st.metric("EtCO₂", f"{etco2:.1f} mmHg")

            with vcol4:
                st.metric("Resp Rate", f"{resp_rate:.0f} /min")
                st.metric("Skin Temp", f"{skin_temp:.1f} °C")

        # Update actions display (Tab 2)
        with actions_container:
            acol1, acol2, acol3 = st.columns(3)

            with acol1:
                st.markdown("##### 💧 Crystalloid")
                st.progress(int((cryst / 400)*100))
                st.write(f"{cryst:.0f} mL/min")
                st.caption(f"Action: {a[0]:.3f}")

            with acol2:
                st.markdown("##### 🩸 Blood")
                st.progress(int((blood / 400)*100))
                st.write(f"{blood:.0f} mL/min")
                st.caption(f"Action: {a[1]:.3f}")

            with acol3:
                st.markdown("##### 💊 Vasopressor")
                st.progress(int((vp / 0.04)*100))
                st.write(f"{vp:.3f} mL/min")
                st.caption(f"Action: {a[2]:.3f}")

            # Show clot fraction
            st.markdown("---")
            st.markdown(f"**Clot Fraction:** {info.get('clot_frac', 0):.3f}")
            st.progress(info.get('clot_frac', 0))

        # Update plot (Tab 3) - every few steps to avoid lag
        if step % 2 == 0 or done:
            df = pd.DataFrame(records)
            fig = create_compact_plot(df, total_reward)
            with plot_placeholder:
                st.pyplot(fig)
            plt.close(fig)

        # Update status
        if done:
            outcome = info.get('o', 'unknown')
            if outcome == 'stabilization':
                status_placeholder.success(f"✅ Patient stabilized after {step} minutes!")
            else:
                status_placeholder.error(f"❌ Episode ended: {outcome}")
        else:
            status_placeholder.info(f"⏳ Treating patient... Cumulative reward: {total_reward:.2f}")

        step += 1

        # Optional delay for visibility
        if st.session_state.get('show_delay', True):
            time.sleep(0.1)

    return records, total_reward, map_violations, info


def create_compact_plot(df, total_reward):
    """Streamlined 3-panel plot for live updates."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # MAP with target zone
    ax1.plot(df["timestep"], df["MAP"], 'b-', linewidth=2, label="MAP")
    ax1.axhline(65, color='green', linestyle='--', alpha=0.5, label="Target")
    ax1.fill_between(df["timestep"], 55, 75, alpha=0.1, color='green')
    ax1.set_ylabel("MAP (mmHg)", fontweight='bold')
    ax1.set_title(f"Live Episode Data | Total Reward: {total_reward:.2f}", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # HR
    ax2.plot(df["timestep"], df["HR"], 'r-', linewidth=2, label="Heart Rate")
    ax2.set_ylabel("HR (bpm)", fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Actions (stacked area)
    ax3.fill_between(df["timestep"], 0, df["cryst_ml_per_min"],
                     step='post', alpha=0.6, label="Crystalloid", color='skyblue')
    ax3.fill_between(df["timestep"], df["cryst_ml_per_min"],
                     df["cryst_ml_per_min"] + df["blood_ml_per_min"],
                     step='post', alpha=0.6, label="Blood", color='darkred')
    ax3.set_ylabel("Infusion (mL/min)", fontweight='bold')
    ax3.set_xlabel("Time (minutes)", fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Hemorrhage RL Demo", layout="wide",
                   page_icon="🏥")

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stProgress > div > div {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Hemorrhage Resuscitation AI Agent")
st.markdown("**Real-time demonstration of RL-based treatment for hemorrhagic shock**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    max_steps = st.slider("Maximum timesteps", 20, 100, 60)
    deterministic = st.checkbox("Deterministic policy", value=True,
                                help="Use deterministic actions (recommended for demo)")
    show_delay = st.checkbox("Add display delay", value=True,
                             help="Slow down updates so you can watch (0.1s per step)")

    # Store in session state so function can access it
    st.session_state['show_delay'] = show_delay

    st.markdown("---")
    st.markdown("### 📚 Model Information")
    st.info("""
    **Low Severity**: PPO  
    **High Severity**: Recurrent PPO (LSTM)

    The gating model automatically classifies severity based on initial blood volume changes.
    """)

    st.markdown("---")
    st.markdown("### 🎯 Target Ranges")
    st.markdown("- **MAP**: 55-75 mmHg (high-sev: 55-65, low-sev: 65-75)")
    st.markdown("- **Shock Index**: < 1.0")

    st.markdown("---")
    st.markdown("### ℹ️ Display Settings")
    if show_delay:
        st.success("✅ Delay enabled: ~6 seconds per episode")
    else:
        st.warning("⚡ Delay disabled: Episodes run at max speed")

# Main area
if 'episode_running' not in st.session_state:
    st.session_state.episode_running = False

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    run_button = st.button('🚀 Start New Episode', type="primary",
                           disabled=st.session_state.episode_running,
                           use_container_width=True)

with col2:
    if st.button('📥 Download Last Episode',
                 disabled='last_episode' not in st.session_state):
        if 'last_episode' in st.session_state:
            csv = st.session_state.last_episode.to_csv(index=False)
            st.download_button("Download CSV", csv, "episode.csv", "text/csv")

with col3:
    if st.button('🔄 Reset'):
        st.session_state.clear()
        st.rerun()

if run_button:
    st.session_state.episode_running = True

    base_env = make_env()
    eval_env = DummyVecEnv([make_env])

    records, total_reward, map_violations, final_info = run_episode_with_live_updates(
        base_env, eval_env, max_steps=max_steps, deterministic=deterministic
    )

    # Save for download
    st.session_state.last_episode = pd.DataFrame(records)
    st.session_state.episode_running = False

    # Final summary
    st.markdown("---")
    st.markdown("## 📊 Episode Summary")

    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Total Reward", f"{total_reward:.2f}")
    with sum_col2:
        st.metric("Duration", f"{len(records)} min")
    with sum_col3:
        outcome = final_info.get('o', 'unknown')
        st.metric("Outcome", outcome)
    with sum_col4:
        st.metric("MAP Violations", map_violations)

    # Confetti if successful!
    if final_info.get('o') == 'stabilization':
        st.balloons()

    st.success("Episode complete! Press 'Download Last Episode' to save data.")

else:
    # Instructions when not running
    st.info("""
    ### 👋 Welcome!

    This demo shows an AI agent treating hemorrhagic shock in real-time.

    **What happens:**
    1. A patient with random characteristics and hemorrhage severity is generated
    2. The gating model classifies severity (high/low)
    3. The appropriate expert agent takes over
    4. You watch the agent make treatment decisions every minute
    5. Episode ends when patient is stabilized or dies

    **Click 'Start New Episode' to begin!**
    """)