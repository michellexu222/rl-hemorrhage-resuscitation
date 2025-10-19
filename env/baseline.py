from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from hemorrhage_env import HemorrhageEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import os
#print(os.getcwd())
torch.manual_seed(2)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.makedirs(parent_dir, exist_ok=True)

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

def make_env():
    env = HemorrhageEnv()
    env = Monitor(env)
    return env

# ----- Environment -----
train_env = DummyVecEnv([make_env])
train_env.seed(2)
# Add normalization (running mean/std for obs)
#train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
train_env = VecNormalize.load(os.path.join(parent_dir, "venv_stats_seed2_1.pkl"), train_env)
train_env.training = True
train_env.norm_reward = True

eval_env = DummyVecEnv([make_env])
#eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
eval_env = VecNormalize.load(os.path.join(parent_dir, "venv_stats_seed2_1.pkl"), eval_env)

# # sync normalization stats between training and eval envs
# eval_env.obs_rms = train_env.obs_rms
# eval_env.ret_rms = train_env.ret_rms
eval_env.training = False
eval_env.norm_reward = False

# ----- Callbacks -----
checkpoint_callback = CheckpointCallback(save_freq=1000, # number of steps (not episodes) between checkpoints
                                         save_path=os.path.join(parent_dir, "models", "checkpoints", "baseline_ppo_seed2_1"),
                                         name_prefix="baseline_ppo_seed2")


eval_callback = EvalCallback(
    eval_env,
    #best_model_save_path=os.path.join(parent_dir, "models", "best"),
    log_path=os.path.join(parent_dir, "models", "eval_logs"),
    eval_freq=2000,            # every 2000 steps
    n_eval_episodes=5,
    deterministic=True
)


log_dir = os.path.join(parent_dir, "ppo_pulse_logs")
#model = PPO("MlpPolicy", train_env, n_steps=512, batch_size=256, n_epochs=10, seed=2, learning_rate=1e-4, verbose=1, tensorboard_log=log_dir)
checkpoint_path = os.path.join(parent_dir, "models", "checkpoints", "baseline_ppo_seed2_1", "baseline_ppo_seed2_15000_steps.zip")

model = PPO.load(checkpoint_path, env=train_env, tensorboard_log=log_dir)
model.learn(total_timesteps=15000, reset_num_timesteps=False, callback=[RewardCallback(), eval_callback, checkpoint_callback], tb_log_name="baseline_ppo")

train_env.save(os.path.join(parent_dir, "venv_stats_seed2_1.pkl"))
model.save(os.path.join(parent_dir, "models", "ppo_baseline_seed2_1"))


