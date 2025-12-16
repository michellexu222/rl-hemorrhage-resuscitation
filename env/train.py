import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python")
sys.path.insert(1, r"C:\Users\michellexu\builds\pulse-engine-conda\install\bin")
sys.path.insert(2, r"C:\Users\michellexu\builds\pulse-engine-conda\Innerbuild\src\python")
import os
import torch
import wandb
import yaml
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from hemorrhage_env import HemorrhageEnv

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
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

# using wandb
def main(config=None):

    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with wandb.init(config=config, sync_tensorboard=True):
        config = wandb.config

        # --- Env Setup ---
        n_envs = 8
        train_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        train_env.seed(42)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
        train_env.training = True
        train_env.norm_reward = True

        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False

        # --- Directories ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        model_dir = os.path.join(parent_dir, "models", "wandb_highsev_rppo")
        os.makedirs(model_dir, exist_ok=True)

        log_dir = os.path.join(parent_dir, "rppo_wandb_logs")
        os.makedirs(log_dir, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            log_path=model_dir,
            eval_freq=2048 // n_envs,
            n_eval_episodes=5,
            deterministic=True
        )

        wandb_callback = WandbCallback(
            model_save_freq=0,
            gradient_save_freq=0,
            verbose=2
        )


        # --- PPO Model ---
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            n_steps=config.n_steps // n_envs,
            batch_size=config.batch_size,
            n_epochs=10,
            seed=42,
            learning_rate=config.learning_rate,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            gamma=config.gamma,
            policy_kwargs={"log_std_init": config.log_std_init},
            verbose=1,
            tensorboard_log=log_dir
        )

        # --- Train ---
        model.learn(
            total_timesteps=config.total_timesteps,
            reset_num_timesteps=True,
            callback=[RewardCallback(), eval_callback, wandb_callback],
            tb_log_name="rppo_wandb_highsev"
        )

        # Save VecNormalize stats & final model
        train_env.save(os.path.join(model_dir, f"vecnorm_{wandb.run.name}.pkl"))
        model.save(os.path.join(model_dir, f"model_{wandb.run.name}.zip"))

if __name__ == "__main__":
    main()


# # -----------------------------
# #    Run script normally
# # -----------------------------
# if __name__ == "__main__":
#     from multiprocessing import freeze_support
#     freeze_support()
#
#     # For manual runs:
#     default_config = {
#         "learning_rate": 3e-5,
#         "clip_range": 0.1,
#         "ent_coef": 0.0,
#         "gamma": 0.995,
#         "gae_lambda": 0.92,
#         "batch_size": 128,
#         "n_steps": 1024,
#         "n_epochs": 10,
#         "n_envs": 8,
#         "total_timesteps": 40000,
#     }
#
#     train(default_config)
