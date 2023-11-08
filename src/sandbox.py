import os
import time
import argparse
import pickle
import yaml

import numpy
import torch

import gymnasium as gym
from gymnasium import spaces

import mujoco_py
from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer

from uhc.khrylib.rl.envs.common.mujoco_env import MujocoEnv
from uhc.utils.config_utils.copycat_config import Config
from uhc.smpllib.smpl_robot import Robot


from numpy import bool

def read_yaml_file(yaml_filepath:str="/home/nhstest/uhc_based_envs/src/config/copycat_old/copycat_5.yml"):
    cfg = yaml.safe_load(open(yaml_filepath, 'r'))
    for key, val in cfg.items():
        print(f"{key} : {val}")
                
        
class country(object):
    def __init__(self, country_name, planet) -> None:
        self.country_name = country_name
        self.planet = planet
        
    def show_info(self,):
        # print(f"{self.country_name} is on planet {self.planet}")
        print("nnonon")
    

class city(country):
    def __init__(self, city_name, country_name, planet) -> None:
        # super().__init__(country_name, planet)
        self.city_name = city_name
        
    def show_info(self):
        super().show_info()
        print(f"and the city name is {self.city_name}")
    
    @property    
    def city_name(self):
        print(f"Get Function: {self._city_name}")
        return self._city_name
    
    @city_name.setter   
    def city_name(self, city_name):
        print(f"The New Name is {city_name}")
        self._city_name = city_name
    
    # city_name = property(get_city_name, set_city_name)
    
    def check_var(self):
        print(f"no us : {self.city_name}")
        print(f"yes us : {self._city_name}")



if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__)) 
    read_yaml_file()
    
    # nordic = country(country_name='Nordic Empire', planet='Sera')
    # # nordic.show_info()
    
    # wf = city(city_name='WinterFront', country_name='Nordic Empire', planet='Sera')
    # # wf.show_info()
    # # print("the city : ", wf.get_city_name())
    # print("Changing City Name")
    # wf.city_name = "newyork"
    # print("Cur City Name : ", wf.city_name)
    # wf.check_var()
    
    