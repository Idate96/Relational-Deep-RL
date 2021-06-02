from stable_baselines3 import SAC, PPO, A2C
from helpers import parallel_boxworlds

if __name__ == '__main__':
  log_dir = 'logs/box_world'
  env = parallel_boxworlds(
      6, max_steps=256, goal_length=2, step_cost=-0.05, num_distractors=0, log_dir=log_dir, num_envs=12
  )
  trained_model = PPO.load("logs/box_world/ppo_split_model_9600000_steps", verbose=1)
  trained_model.set_env(env)
  trained_model.learn(10000, reset_num_timesteps=False)