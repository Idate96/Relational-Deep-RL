from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from net import RelationalNet
import torch.nn as nn
import torch as th
import gym
import numpy as np

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


class SimpleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=256)
        # the network does not contain the final projection
        # with mlp_depth = 4, set net_arch = []
        # else put the mlp layers into a custom network
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(2, 2), stride=1),
            nn.BatchNorm2d(12),
            nn.GELU(),
            nn.Conv2d(12, 24, kernel_size=(2, 2), stride=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Flatten()
        )
        # for some reason the standard is 256 
        # self.linear = nn.Sequential(nn.Linear(864, 512), 
        #                             nn.GELU(),
        #                             nn.Linear(512, 256),
        #                             nn.GELU(), 
        #                             nn.Linear(256, 256),
        #                             nn.GELU())
        self.linear = nn.Sequential(nn.Linear(3456, 512), 
                            nn.GELU(),
                            nn.Linear(512, 256),
                            nn.GELU(), 
                            nn.Linear(256, 256),
                            nn.GELU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv(observations)
        print(x.shape)
        return self.linear(x)

class SimpleExtractorDict(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=256)
        # the network does not contain the final projection
        # with mlp_depth = 4, set net_arch = []
        # else put the mlp layers into a custom network
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(2, 2), stride=1),
            nn.BatchNorm2d(12),
            nn.GELU(),
            nn.Conv2d(12, 24, kernel_size=(2, 2), stride=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Flatten()
        )
        # for some reason the standard is 256 
        # self.linear = nn.Sequential(nn.Linear(864, 512), 
        #                             nn.GELU(),
        #                             nn.Linear(512, 256),
        #                             nn.GELU(), 
        #                             nn.Linear(256, 256),
        #                             nn.GELU())
        self.linear = nn.Sequential(nn.Linear(3456, 512), 
                            nn.GELU(),
                            nn.Linear(512, 256),
                            nn.GELU(), 
                            nn.Linear(256, 256),
                            nn.GELU())

    def forward(self, observations) -> th.Tensor:
        image = observations['image']
        print("image shape ", image.shape)
        x_image = self.conv(image)
        x_state = th.cat((th.squeeze(observations['agent_orientation']), observations['agent_carrying']))
        x = th.cat((x_image, x_state))
        return self.linear(x)



class DeeperExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=64)
        # the network does not contain the final projection
        # with mlp_depth = 4, set net_arch = []
        # else put the mlp layers into a custom network
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(2, 2), stride=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(2, 2), stride=1),
            nn.GELU(),
            nn.Flatten()
        )
        # for some reason the standard is 256 
        # self.linear = nn.Sequential(nn.Linear(48, 64), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv(observations)
        print(x.shape)
        return x