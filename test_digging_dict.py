import os
from random import seed
import numpy as np
from gym.envs.registration import make
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import MlpPolicy
import gym
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    DummyVecEnv,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from custom_callbacks import VideoRecorderCallback, TensorboardCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import tensorflow as tf
from helpers import make_boxworld, make_env, parallel_boxworlds, parallel_worlds
from net import RelationalNet
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from extractors import RelationalNet, DeeperExtractor, SimpleExtractor, SimpleExtractorDict, CustomCombinedExtractor
import heightgrid  # register the environment

# checkpoint (model and replay buffer): check
# logging (personalized): -
# on custom environment




if __name__ == "__main__":
    log_dir = "./logs/heightgrid/ppo/digging_8x8/dict"
    os.makedirs(log_dir, exist_ok=True)
    size = 8
    num_digging_pts = 4
    env_id = "HeightGrid-RandomTargetHeight-v0"
    env = parallel_worlds(env_id, log_dir=log_dir, flat_obs=False, num_envs=16, 
                          size=size, step_cost=-0.01, num_digging_pts=num_digging_pts, max_steps=1024)

    eval_env = make_env(env_id, log_dir=log_dir, seed=24, flat_obs=False, 
                        size=size, step_cost=-0.01, num_digging_pts=num_digging_pts, max_steps=1024)()
    # figure, ax = eval_env.render()
    # plt.plot(figure)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[256, dict(pi=[256], vf=[256])]
    )


    # with steps 2058 * num_envs

    checkpoint_callback = CheckpointCallback(
        save_freq=200000, save_path=log_dir, name_prefix="ppo_dig"
    )

    # Separate evaluation env

    # eval_env.render('human')
    # check_env(eval_env)
    # print("Created eval env")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "/best_model",
        log_path=log_dir + "/results",
        eval_freq=20000,
    )

    # not saving
    video_recorder = VideoRecorderCallback(eval_env, render_freq=20000)
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    continue_training = 1
    
    if continue_training:
      model_log_dir = 'logs/heightgrid/ppo/digging_8x8/dict/ppo_dig_308815872_steps'
      env.reset()
      model = PPO.load(model_log_dir, env=env)
      
      model.learn(
        20e7,
        callback=callbacks,
        tb_log_name="ppo_dig",
        reset_num_timesteps=False,
    )

    
    else:

      model = PPO(
          ActorCriticPolicy,
          env,
          gamma=0.999,
          batch_size=2048,
          n_steps=1024,  
          n_epochs=4,
          ent_coef=0.001,
          learning_rate=0.001, 
          policy_kwargs=policy_kwargs,
          verbose=1,
          create_eval_env=True,
          tensorboard_log=log_dir,
      )
    # model.load('logs/heightgrid/ppo/digging_8x8/ppo_goal_target_9600000_steps', env=env)
      model.learn(
          20e7,
          callback=callbacks,
          tb_log_name="ppo_dig",
          reset_num_timesteps=True,
      )

    # save the model
    model.save(log_dir + "/ppo_dig_model")
    # now save the replay buffer too
    # model.save_replay_buffer("sac_replay_buffer")

    # # load the model
    # loaded_model = PPO.load("ppo_boxworld", env=env)
    # # load it into the loaded_model
    # # loaded_model.load_replay_buffer("sac_replay_buffer")

    # loaded_model.learn(
    #     total_timesteps=7000, tb_log_name="second", reset_num_timesteps=False
    # )

    # # Retrieve the environment
    # env = model.get_env()

    # # Evaluate the policy
    # mean_reward, std_reward = evaluate_policy(
    #     model.policy, eval_env, n_eval_episodes=10, deterministic=True
    # )
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # obs = eval_env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = eval_env.step(action)
    #     env.render("human")
