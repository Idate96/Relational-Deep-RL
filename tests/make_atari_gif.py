from os import name
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
import imageio
import numpy as np

if __name__ == "__main__":
    log_dir = "logs/atari/pong"
    env_id = "PongNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)

    env.reset()
    video_length = 1280

    model_path = "logs/atari/pong/ppo_11200000_steps"
    model = PPO.load(model_path, env=env)
    images = []

    obs = env.reset()
    img = env.render(mode="rgb_array")
    for i in range(video_length):
        images.append(img)
        action, state_ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        img = env.render(mode="rgb_array")

    imageio.mimsave(
        log_dir + "/videos/" + env_id + ".gif", [np.array(img) for i, img in enumerate(images)]
    )
