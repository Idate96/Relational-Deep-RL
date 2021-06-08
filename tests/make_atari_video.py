from os import name
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder


if __name__ == "__main__":
    log_dir = "logs/atari/pong"
    env_id = "PongNoFrameskip-v4"
    env = make_atari_env(env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)

    env.reset()
    video_length = 128
    env = VecVideoRecorder(
        env,
        log_dir + "/videos",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="agent-{}".format(env_id),
    )
    model_path ='logs/atari/pong/ppo_11200000_steps'
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    for i in range(video_length):
      action, state_ = model.predict(obs, deterministic=True)
      obs, rewards, done, info = env.step(action)
    env.close()