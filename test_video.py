from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import os

# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ["DISPLAY"] = ":1"


def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


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
from helpers import make_boxworld, parallel_boxworlds
from net import RelationalNet
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from extractors import RelationalNet, DeeperExtractor, SimpleExtractor

# checkpoint (model and replay buffer): check
# logging (personalized): -
# on custom environment
# tensorflow integration: check


if __name__ == "__main__":
    log_dir = "./logs/box_world/"
    os.makedirs(log_dir, exist_ok=True)

    # tf.debugging.experimental.enable_dump_debug_info("./logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    # Create the model, the training environment
    # and the test environment (for evaluation)
    env_id = "Boxworld"
    env = make_boxworld(
        6,
        max_steps=256,
        goal_length=2,
        step_cost=-0.05,
        num_distractors=0,
        log_dir=log_dir,
        seed=24
    )()

    eval_env = VecVideoRecorder(VecTransposeImage(
        DummyVecEnv(
            [
                lambda: make_boxworld(
                    6,
                    max_steps=256,
                    step_cost=-0.05,
                    goal_length=2,
                    num_distractors=0,
                    log_dir=log_dir,
                    seed=24,
                )()
            ]
        )
    ), video_folder='logs/box_world/videos',record_video_trigger=lambda step: step % (256 * 2) == 0, video_length=256))

    # figure, ax = eval_env.render()
    # plt.plot(figure)

    policy_kwargs = dict(
        features_extractor_class=DeeperExtractor,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )

    model = PPO(
        ActorCriticPolicy,
        env,
        gamma=1,
        batch_size=256,
        n_steps=256,
        n_epochs=4,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        create_eval_env=True,
        tensorboard_log=log_dir,
    )
    # with steps 2058 * num_envs

    checkpoint_callback = CheckpointCallback(
        save_freq=200000, save_path=log_dir, name_prefix="ppo_split_model"
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
    max_min_callback = TensorboardCallback(log_dir)
    callbacks = CallbackList(
        [eval_callback, checkpoint_callback, video_recorder, max_min_callback]
    )

    # Create the callback list
    # Evaluate the model every 1000 steps on 5 test episodes
    # and save the evaluation to the "logs/" folder
    model.learn(10000000, callback=callbacks, tb_log_name="ppo_run_split")

    # save the model
    model.save("ppo_boxworld_n_test")
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
