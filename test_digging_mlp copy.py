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
from extractors import RelationalNet, DeeperExtractor, SimpleExtractor
import heightgrid  # register the environment

# checkpoint (model and replay buffer): check
# logging (personalized): -
# on custom environment
# tensorflow integration: check




if __name__ == "__main__":
    log_dir = "./logs/heightgrid/ppo/digging_5x5"
    os.makedirs(log_dir, exist_ok=True)
    size = 5
    num_digging_pts = 1
    env_id = "HeightGrid-RandomTargetHeight-v0"
    env = parallel_worlds(env_id, log_dir=log_dir, flat_obs=True, num_envs=16, size=size, num_digging_pts=num_digging_pts, max_steps=2048)

    eval_env = make_env(env_id, log_dir=log_dir, seed=24, flat_obs=True, size=size, num_digging_pts=num_digging_pts)()
    # figure, ax = eval_env.render()
    # plt.plot(figure)

    policy_kwargs = dict(net_arch=[128, dict(pi=[128], vf=[128])])

    model = PPO(
        "MlpPolicy",
        env,
        gamma=1,
        batch_size=1024,
        n_steps=2048,  
        n_epochs=4,
        ent_coef=0.001,
        learning_rate=0.0003, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        create_eval_env=True,
        tensorboard_log=log_dir,
    )
    # with steps 2058 * num_envs

    checkpoint_callback = CheckpointCallback(
        save_freq=200000, save_path=log_dir, name_prefix="ppo_goal_target"
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

    # Create the callback list
    # Evaluate the model every 1000 steps on 5 test episodes
    # and save the evaluation to the "logs/" folder
    model.learn(
        10e7,
        callback=callbacks,
        tb_log_name="ppo_goal",
        reset_num_timesteps=False,
    )

    # save the model
    model.save(log_dir + "/ppo_goal_model")
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

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model.policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    obs = eval_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        env.render("human")
