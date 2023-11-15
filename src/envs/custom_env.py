import os
import sys
import argparse

from typing import Any, List

import numpy as np
import torch

import gym
from gym import spaces

from embodiedpose.envs.humanoid_kin_res import HumanoidKinEnvRes



class CustomEnv(gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, custom_env:HumanoidKinEnvRes):
        super(CustomEnv, self).__init__()
        # # Define action and observation space
        # # They must be gym.spaces objects
        # # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        
        # get custom env
        self.custom_env = custom_env
        
        # set action space and observation space
        self.action_space = custom_env.action_space
        self.observation_space = custom_env.observation_space
        
        # width and height for render

    def step(self, action):
        return self.custom_env.step(action)

    def reset(self):
        return self.custom_env.reset()

    def render(self, mode='human'):
        self.custom_env.render(mode=mode)
        
    def close (self):
        self.custom_env.close()
    
    
    
    
if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))