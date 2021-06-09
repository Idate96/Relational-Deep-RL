import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == "__main__":
  log_dir = "logs/atari/pong"
  env_id = "PongNoFrameskip-v4"
  num_envs = 2
  env = make_atari_env(env_id, n_envs=num_envs, seed=0, monitor_dir=log_dir)
  env = VecFrameStack(env, n_stack=4)

  model = PPO(
      "CnnPolicy",
      env,
      batch_size=256,
      n_steps=128,
      ent_coef=0.01,
      n_epochs=4,
      verbose=1,
  )
  checkpoint_callback = CheckpointCallback(
      save_freq=200000, save_path=log_dir, name_prefix="ppo"
  )
  model.learn(
      1e7, callback=checkpoint_callback
  )


  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()