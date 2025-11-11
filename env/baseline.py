import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# # reduce thread overhead (Windows)
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
# os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import torch
# torch.set_num_threads(1)
# torch.backends.mkldnn.enabled = False

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from hemorrhage_env import HemorrhageEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

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

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    torch.manual_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.makedirs(parent_dir, exist_ok=True)

    # ----- Environment -----
    n_envs = 8
    train_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    train_env.seed(42)
    #train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    train_env = VecNormalize.load(os.path.join(parent_dir, "venv_stats_14.pkl"), train_env)
    train_env.training = True
    train_env.norm_reward = True

    eval_env = DummyVecEnv([make_env])
    #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env = VecNormalize.load(os.path.join(parent_dir, "venv_stats_14.pkl"), eval_env)
    #eval_env.obs_rms = train_env.obs_rms
    #eval_env.ret_rms = train_env.ret_rms
    eval_env.training = False
    eval_env.norm_reward = False

    # ----- Callbacks -----
    checkpoint_callback = CheckpointCallback(
        save_freq=2048//n_envs,
        save_path=os.path.join(parent_dir, "models", "checkpoints", "baseline_ppo_14_clotting"),
        name_prefix="ppo_14_clotting"
    )

    eval_callback = EvalCallback(
        eval_env,
        log_path=os.path.join(parent_dir, "models", "eval_logs"),
        eval_freq=2048//n_envs,
        n_eval_episodes=5,
        deterministic=True
    )

    log_dir = os.path.join(parent_dir, "ppo_pulse_logs")
    #model = PPO("MlpPolicy", train_env, n_steps=512//n_envs, batch_size=256, n_epochs=10, seed=42, learning_rate=1e-4, verbose=1, tensorboard_log=log_dir)
    checkpoint_path = os.path.join(parent_dir, "models", "ppo_baseline_14.zip")
    model = PPO.load(checkpoint_path, env=train_env, tensorboard_log=log_dir)
    model.learn(total_timesteps=20000, reset_num_timesteps=True, callback=[RewardCallback(), eval_callback, checkpoint_callback], tb_log_name="baseline_ppo_14_clotting")

    train_env.save(os.path.join(parent_dir, "venv_stats_ppo_14_clotting.pkl"))
    model.save(os.path.join(parent_dir, "models", "ppo_14_clotting"))

