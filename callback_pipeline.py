import os
import numpy as np
from gym.envs.registration import make
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from custom_callbacks import VideoRecorderCallback, TensorboardCallback
import tensorflow as tf


# checkpoint (model and replay buffer)
# logging (personalized)
# on custom environment
# tensorflow integration


if __name__ == "__main__":
  log_dir = "./logs/sac_pendulum"
  # tf.debugging.experimental.enable_dump_debug_info("./logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
  # Create the model, the training environment
  # and the test environment (for evaluation)
  env_id = "CartPole-v1"
  env = make_vec_env(env_id, n_envs=1, seed=0, monitor_dir=log_dir)
  model = PPO(
      "MlpPolicy",
      env,
      verbose=1,
      learning_rate=1e-3,
      create_eval_env=True,
      tensorboard_log="./ppo_cartpole/",
  )

  checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir)
  # Separate evaluation env
  eval_env = gym.make("CartPole-v1")
  eval_callback = EvalCallback(
      eval_env,
      best_model_save_path="./logs/best_model",
      log_path="./logs/results",
      eval_freq=1000,
  )

  video_recorder = VideoRecorderCallback(env, render_freq=1000)
  max_min_callback = TensorboardCallback(log_dir)
  callbacks = CallbackList(
      [eval_callback, checkpoint_callback, video_recorder, max_min_callback]
  )

  # Create the callback list
  # Evaluate the model every 1000 steps on 5 test episodes
  # and save the evaluation to the "logs/" folder
  model.learn(6000, callback=callbacks, tb_log_name="first_run")

  # save the model
  model.save("ppo_cartpole")
  # now save the replay buffer too
  model.save_replay_buffer("sac_replay_buffer")

  # load the model
  loaded_model = PPO.load("ppo_cartpole")
  # load it into the loaded_model
  loaded_model.load_replay_buffer("sac_replay_buffer")

  loaded_model.learn(
      total_timesteps=10000, tb_log_name="second", reset_num_timesteps=False
  )

  # Retrieve the environment
  env = model.get_env()

  # Evaluate the policy
  mean_reward, std_reward = evaluate_policy(
      loaded_model.policy, env, n_eval_episodes=10, deterministic=True
  )
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, dones, info = env.step(action)
      env.render()
