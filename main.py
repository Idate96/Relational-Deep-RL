import os
import gym
import numpy as np
import gym_boxworld
from gym_boxworld.envs.boxworld_env import BoxworldEnv, RandomBoxworldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecVideoRecorder,
    DummyVecEnv,
)
import argparse
import PIL
import imageio
from stable_baselines3 import A2C
import torch.nn as nn
import torch as th


def make_env(seed: int, log_dir: str, random: bool = False) -> Callable[[], gym.Env]:
    """Create custom minigrid environment

    Args:
        env_name (str): name of the minigrid

    Returns:
        init: function that when called instantiate a gym environment
    """

    def init():

        env = BoxworldEnv()
        # default is partially observable
        env.seed(seed)
        env = Monitor(env, log_dir)
        # env = DummyVecEnv([lambda: env])
        # env = VecNormalize(env)

        return env

    return init


def set_env(config, log_dir):
    env = SubprocVecEnv(
        [make_env(i, log_dir, random=True) for i in range(config.num_cpu)]
    )
    return env


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-num_cpu", default=12, type=int, help="whether use frame_stack, default=False"
    )
    config = parser.parse_args()

    log_dir = "tmp/"
    video_dir = log_dir + "video/"
    video_length = 30

    os.makedirs(log_dir, exist_ok=True)
    envs = set_env(config, log_dir)
    env = make_env(0, log_dir)()
    # env = VecVideoRecorder(env, video_dir,
    #                      record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                      name_prefix="random-agent-{}".format("boxworld"))

    # env.reset()

    # for _ in range(1):
    #   obs, rewards, dones, info = env.step(np.random.randint(0, 3))
    #   env.render()

    # env.close()

    model = PPO(
        "CnnPolicy",
        env=envs,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
        ),
        verbose=1,
    ).learn(1000000)

    # model.save("ppo_boxworld")
    # model = PPO.load("ppo_boxworld", env=envs)
    # model.learn(1000000, reset_num_timesteps=True)
    model.save("ppo_boxworld")

    # env = DummyVecEnv([lambda: )])

    # env = VecVideoRecorder(env, video_dir,
    #                      record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                      name_prefix="random-agent-{}".format("boxworld"))

    # env.reset()

    # for _ in range(1):
    #   obs, rewards, dones, info = env.step(np.random.randint(0, 3))
    #   env.render()

    # env.close()

    obs = env.reset()
    images = []
    img = env.render(mode="return")
    for i in range(35):
        # img = PIL.Image.fromarray(img)
        # img = img.resize((128, 128), PIL.Image.ANTIALIAS)
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        img = env.render(mode="return")
        if i % 5 == 0:
            env.reset()
            img = env.render(mode="return")

    imageio.mimsave("boxworld_a2c.gif", images, fps=2)
