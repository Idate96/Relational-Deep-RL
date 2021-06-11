from os import name
import gym
from gym.envs.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
import imageio
import numpy as np
from helpers import make_env, parallel_worlds

if __name__ == "__main__":
    log_dir = "logs/heightgrid/ppo/digging_16x16/dict_mask"

    size = 16
    num_digging_pts = 2
    env_id = "HeightGrid-RandomTargetHeight-v0"
    env = parallel_worlds(env_id, log_dir=log_dir, flat_obs=False, num_envs=1, 
                          size=size, step_cost=-0.001, mask=True, num_digging_pts=num_digging_pts, max_steps=1024)
    env.reset()
    video_length = 2048
    model_path = "logs/heightgrid/ppo/digging_16x16/dict_mask/mask_pts2_3200000_steps"
    model = PPO.load(model_path, env=env)

    num_gifs = 5
    for gif_id in range(num_gifs):
      images = []
      obs = env.reset()
      img = env.render(mode="rgb_array")
      for i in range(video_length):
          images.append(img)
          action, state_ = model.predict(obs, deterministic=True)
          obs, rewards, done, info = env.step(action)
          img = env.render(mode="rgb_array")
          if done:
            break

      imageio.mimsave(
          log_dir + "/videos/" + env_id + "steps" + "_" + str(i) + "_" + str(gif_id) + ".gif", [np.array(img) for i, img in enumerate(images)]
      )
