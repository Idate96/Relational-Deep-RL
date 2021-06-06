from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from net import RelationalNet
import torch.nn as nn
import torch as th
import gym
import numpy as np
from torch import autograd


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
            nn.Flatten(),
        )
        # for some reason the standard is 256
        # self.linear = nn.Sequential(nn.Linear(864, 512),
        #                             nn.GELU(),
        #                             nn.Linear(512, 256),
        #                             nn.GELU(),
        #                             nn.Linear(256, 256),
        #                             nn.GELU())
        self.linear = nn.Sequential(
            nn.Linear(3456, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

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
            nn.Flatten(),
        )
        # for some reason the standard is 256
        # self.linear = nn.Sequential(nn.Linear(864, 512),
        #                             nn.GELU(),
        #                             nn.Linear(512, 256),
        #                             nn.GELU(),
        #                             nn.Linear(256, 256),
        #                             nn.GELU())
        self.linear = nn.Sequential(
            nn.Linear(3456, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

    def forward(self, observations) -> th.Tensor:
        image = observations["image"]
        print("image shape ", image.shape)
        x_image = self.conv(image)
        x_state = th.cat(
            (
                th.squeeze(observations["agent_orientation"]),
                observations["agent_carrying"],
            )
        )
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
            nn.Flatten(),
        )
        # for some reason the standard is 256
        # self.linear = nn.Sequential(nn.Linear(48, 64), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv(observations)
        print(x.shape)
        return x


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=2),
                    nn.ReLU(),
                )

                total_concat_size += self.feature_size(subspace.shape)
                
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def feature_size(self, shape):
        return self.features(autograd.Variable(th.zeros(1, shape))).view(1, -1).size(1)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
