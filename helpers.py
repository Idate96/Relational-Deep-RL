from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_boxworld.envs.boxworld_env import BoxworldEnv, RandomBoxworldEnv
from typing import Callable
import gym
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder



def make_boxworld(n:int, max_steps:int, goal_length:int, step_cost:int, num_distractors:int, seed:int, log_dir: str, monitor=True, random: bool = False) -> Callable[[], gym.Env]:
    """Create custom minigrid environment

    Args:
        env_name (str): name of the minigrid

    Returns:
        init: function that when called instantiate a gym environment
    """

    def init():
        if random:
          env = RandomBoxworldEnv(n=n, max_steps=max_steps, step_cost = step_cost, goal_length=goal_length, num_distractor=num_distractors, )
        else:
          env = BoxworldEnv(n=n, max_steps=max_steps, step_cost = step_cost, goal_length=goal_length, num_distractor=num_distractors)
        env.seed(seed)
        # when using parallel environments only allow 1 env to log 
        if monitor:
          env = Monitor(env, log_dir)
          # env = DummyVecEnv(env)
          # env = VecVideoRecorder(env, video_folder=log_dir + "/videos", 
          #       record_video_trigger=lambda step: step % (max_steps * 2) == 0, video_length=256)
        # env = DummyVecEnv(env)
        return env

    return init


def parallel_boxworlds(n:int, max_steps, goal_length:int, num_distractors:int, step_cost:float, log_dir, num_envs):
    env = SubprocVecEnv(
        [make_boxworld(n=n, max_steps=max_steps, goal_length=goal_length, step_cost=step_cost, num_distractors=num_distractors, seed=i, monitor=i==0, log_dir=log_dir, random=False) for i in range(num_envs)]
    )
    return env


def conv2d_size_out(size, kernel_size=2, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1
    


