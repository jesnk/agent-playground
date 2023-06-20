# save model
#model.save("sac_fetch_reach")
# load model
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
from jesnk_utils.rgb_to_video import RGB2VIDEO
import cv2

model = SAC.load("sac_fetch_reach_100000")

# test model
rgb_to_video = RGB2VIDEO()

env = MujocoPyFetchReachEnv(reward_type='dense')
env.render_mode = 'rgb_array'
#env = Monitor(env_eval, log_dir)
env = TimeLimit(env, max_episode_steps=100)
env = DummyVecEnv([lambda: env])
env.render_mode = 'rgb_array'
obs = env.reset()
frames = []
episode_step = 0
episode_num = 0
cum_reward = 0
for i in range(1,501):    
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    cum_reward += rewards[0]
    episode_step += 1
    frame = env.render()
    # insert infos into frame, like episode number, episode step, reward, etc.
    # convert frame to cv2 image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.putText(frame, f'episode: {episode_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, f'episode step: {episode_step}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, f'reward: {rewards[0]:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, f'cumulative reward: {cum_reward:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    frames.append(frame)
    if dones[0]:
        obs = env.reset()
        episode_step = 0
        episode_num += 1   
        cum_reward = 0 

        
rgb_to_video.set_frames(frames)
rgb_to_video.set_fps(5)
rgb_to_video.save(path=f'{i}_test.gif',mode='gif')
print(f'episode {i} done')
frames = []
rgb_to_video.container.clear()

# save model