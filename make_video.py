import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from helpers import make_env

log_dir = "./logs/heightgrid/ppo/digging_5x5"
video_folder = log_dir + "/videos"
size = 5
num_digging_pts = 1
env_id = "HeightGrid-RandomTargetHeight-v0"
video_length = 100
# without the dummy vec env a weird recursion depth error appears 
env = DummyVecEnv([make_env(env_id, log_dir=log_dir, seed=24, flat_obs=False, monitor=False, size=size, num_digging_pts=num_digging_pts)])

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="random-agent-{}".format(env_id))

env.reset()
for _ in range(video_length + 1):
  action = [env.action_space.sample()]
  obs, _, _, _ = env.step(action)
# Save the video
env.close()