"""
We test here the implementation of the transformer model with a simplified version of the environment
"""
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import constant_fn
import gym
from net import RelationalNet
import torch as th
import torch.nn as nn

# local imports
from helpers import make_boxworld, parallel_boxworlds


class RelationalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=256)
        # the network does not contain the final projection
        # with mlp_depth = 4, set net_arch = []
        # else put the mlp layers into a custom network
        self.net = RelationalNet(
            input_size=8,
            mlp_depth=4,
            depth_transformer=2,
            heads=2,
            baseline=False,
            recurrent_transformer=True,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


if __name__ == "__main__":
    log_dir = "tmp/"
    video_dir = log_dir + "video/"

    os.makedirs(log_dir, exist_ok=True)
    envs = parallel_boxworlds(n=6, goal_length=3, num_distractors=1, log_dir=log_dir, num_envs=12)
    env = make_boxworld(n=6, goal_length=3, num_distractors=0, seed=0, log_dir=log_dir)()

    policy_kwargs = dict(
        features_extractor_class=RelationalExtractor,
        net_arch=[],
    )

    # model = PPO(ActorCriticPolicy, envs, policy_kwargs=policy_kwargs, verbose=1)
    model = PPO.load('relational_net', env=envs)
    model.learn(1000000, reset_num_timesteps=True)
    model.save('relational_net_test')