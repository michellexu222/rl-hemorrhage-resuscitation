import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import torch
from multiprocessing import freeze_support

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from hemorrhage_env import HemorrhageEnv
from baseline import RewardCallback


def make_env():
    env = HemorrhageEnv()
    env = Monitor(env)
    return env

if __name__ == '__main__':
    freeze_support()
    torch.manual_seed(7)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.makedirs(parent_dir, exist_ok=True)

    # ---------- Training Environment ----------
    n_envs = 8 # number of parallel environments
    train_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )
    train_env.training = True
    train_env.norm_reward = True

    # evaluation environment
    eval_env = DummyVecEnv([make_env])
    # eval_env = VecNormalize(
    #     eval_env,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_obs=10.0
    # )
    eval_env = VecNormalize.load(os.path.join(parent_dir, "venv_stats_sac_1.pkl"), eval_env)
    # eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    # callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2048//n_envs,  # save periodically
        save_path=os.path.join(parent_dir, "models", "checkpoints", "baseline_sac_1"),
        name_prefix="baseline_sac_1"
    )

    eval_callback = EvalCallback(
        eval_env,
        log_path=os.path.join(parent_dir, "models", "eval_logs"),
        eval_freq=2048//n_envs,  # eval every 10k steps total (across all envs)
        n_eval_episodes=5,
        deterministic=True
    )

    log_dir = os.path.join(parent_dir, "sac_pulse_logs")

    # for training a new model:
    # model = SAC(
    #     "MlpPolicy",
    #     train_env,
    #     learning_rate=1e-4,  # same as ppo
    #     buffer_size=40000,  # replay buffer
    #     learning_starts=1000,  # start updates after some data
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=4,  # collect 1 step per env before an update
    #     gradient_steps=8,  # number of gradient steps per train_freq
    #     ent_coef="auto",  # auto entropy tuning
    #     target_update_interval=1,
    #     use_sde=False,
    #     seed=7,
    #     verbose=1,
    #     tensorboard_log=log_dir
    # )

    # continue from previously saved checkpoint:
    checkpoint_path = os.path.join(parent_dir, "models", "sac_baseline_1.zip")
    model = SAC.load(checkpoint_path, env=train_env, tensorboard_log=log_dir)

    # training
    model.learn(
        total_timesteps=20000,
        log_interval=10,
        reset_num_timesteps=False,
        callback=[RewardCallback(), eval_callback, checkpoint_callback],
        tb_log_name="baseline_sac"
    )

    train_env.save(os.path.join(parent_dir, "venv_stats_sac_1.pkl"))
    model.save(os.path.join(parent_dir, "models", "sac_baseline_1"))
    model.save_replay_buffer("sac_buffer_1.pkl")