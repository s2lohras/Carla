# import gymnasium as gym
# from gym.envs.registration import register
from gymnasium.envs.registration import register

register(
    id='CarlaEnv-v1',
    entry_point='envs.Carla_env:CarlaEnv',
)