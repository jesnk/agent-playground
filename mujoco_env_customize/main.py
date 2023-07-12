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
from gym_robotics.envs.fetch.push import MujocoPyFetchPushEnv
from gym.wrappers import TimeLimit
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import datetime
log_dir = "./tb_log/"

goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE


# init mujoco fetch enviroenment
env_name = 'FetchReach'

if env_name == 'FetchReach':
    env_class = MujocoPyFetchReachEnv
elif env_name == 'FetchPush':
    env_class = MujocoPyFetchPushEnv
else :
    raise ValueError(f"env_name: {env_name} is not supported")

model_name = 'SAC'


max_steps = 2_000_000
reward_type = 'sparse'
action_scale = 0.1

time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#distance_threshold = 0.05
config = {
    "policy_type": model_name,
    "total_timesteps": max_steps,
    "env_name": env_name,
    "reward_type": reward_type,
    "max_steps": max_steps,
    "action_scale": action_scale,
    "timestemp": time,
}

name = f"{config['env_name']}-{config['policy_type']}-{config['reward_type']}"
run = wandb.init(
    project="sb3",
    name= name,
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional 
)


# init mujoco fetch environment
env = env_class(reward_type=config['reward_type'], max_episode_steps=200, action_scale=config['action_scale'])
env = Monitor(env, log_dir)
env = TimeLimit(env, max_episode_steps=100)
env = DummyVecEnv([lambda: env])

env_eval = env_class(reward_type=config['reward_type'],max_episode_steps=200, action_scale=config['action_scale'])
env_eval = Monitor(env_eval, log_dir)
env_eval = TimeLimit(env_eval, max_episode_steps=100)
env_eval = DummyVecEnv([lambda: env_eval])

env.render_mode = 'rgb_array'
# wrap environment
# init model

if model_name == 'SAC-HER':
    model = SAC(MlpPolicy, env, verbose=1, 
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,),
            device='cuda',wandb_log=True)

elif model_name == 'SAC':
    model = SAC(MlpPolicy, env, verbose=1,
                device='cuda',wandb_log=True)
else :
    raise ValueError(f"model_name: {model_name} is not supported")

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

model.save(f"./checkpoint/{name}-{time}")
