# train with SAC, stable baseline3
import stable_baselines3
from stable_baselines3 import SAC, PPO
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
# pip install gym-robotics
import matplotlib.pyplot as plt
from gym_robotics.envs.fetch.reach import MujocoPyFetchReachEnv
from gym.wrappers import TimeLimit


# init mujoco fetch enviroenment
env = MujocoPyFetchReachEnv()

log_dir = "./sac_fetch_reach_tensorboard/"
max_steps = 100_000
reward_type = 'sparse'
#distance_threshold = 0.05
config = {
    "policy_type": "PPO",
    "total_timesteps": max_steps,
    "env_name": "FetchReach",
    "reward_type": reward_type,
}
run = wandb.init(
    project="sb3",
    name=f"{config['policy_type']}-{config['env_name']}",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


# init mujoco fetch environment
# init mujoco fetch environment
env = MujocoPyFetchReachEnv(reward_type=config['reward_type'])
env = Monitor(env, log_dir)
#env = TimeLimit(env, max_episode_steps=100)
env = DummyVecEnv([lambda: env])

env_eval = MujocoPyFetchReachEnv(reward_type=config['reward_type'])
env_eval = Monitor(env_eval, log_dir)
#env_eval = TimeLimit(env_eval, max_episode_steps=100)
env_eval = DummyVecEnv([lambda: env_eval])

env.render_mode = 'rgb_array'
# wrap environment
# init model
model = SAC(MlpPolicy, env, verbose=1, 
            device='cuda',wandb_log=True)
#model = PPO(MlpPolicy, env, verbose=1,
#            device='cuda')

# train model
model.learn(total_timesteps=max_steps, 
            log_interval=10, 
            tb_log_name="sac_fetch_reach", 
            reset_num_timesteps=False, 
            eval_freq=100, 
            n_eval_episodes=20,
            eval_log_path="sac_fetch_reach_eval",
            eval_env=env_eval,
            )

model.save(f"sac_fetch_reach_{max_steps}")