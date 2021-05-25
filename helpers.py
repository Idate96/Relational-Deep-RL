from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_boxworld.envs.boxworld_env import BoxworldEnv, RandomBoxworldEnv
from typing import Callable
import gym


def make_boxworld(seed: int, log_dir: str, random: bool = False) -> Callable[[], gym.Env]:
    """Create custom minigrid environment

    Args:
        env_name (str): name of the minigrid

    Returns:
        init: function that when called instantiate a gym environment
    """

    def init():
        if random:
          env = RandomBoxworldEnv()
        else:
          env = BoxworldEnv()
        env.seed(seed)
        env = Monitor(env, log_dir)
        return env

    return init


def parallel_boxworlds(log_dir, num_envs):
    env = SubprocVecEnv(
        [make_boxworld(i, log_dir, random=True) for i in range(num_envs)]
    )
    return env
