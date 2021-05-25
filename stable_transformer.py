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
    envs = parallel_boxworlds(log_dir, num_envs=12)
    env = make_boxworld(0, log_dir)()

    policy_kwargs = dict(
        features_extractor_class=RelationalExtractor,
        net_arch=[],
    )


    # model setup tries to isntatiate the class policy
    # but here we have it already instatiated
    # policy = ActorCriticPolicy(
    #     env.observation_space,
    #     env.action_space,
    #     lr_schedule=constant_fn(1e-5),
    #     net_arch=[],  # identity
    #     features_extractor_class=RelationalExtractor,
    #     ortho_init=False,
    # )

    model = A2C(ActorCriticPolicy, envs, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(100000)
    model.save('relatinal_net')