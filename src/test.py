import os
import argparse
import pickle

import numpy as np
import torch        

from uhc.utils.flags import flags

from embodiedpose.utils.video_pose_config import Config

from envs.embodiedpose_HumanoidKinEnvRes import Custom_HumanoidKinEnvRes
from envs.custom_env import CustomEnv

def get_config(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="tcn_voxel_4_5")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="/home/nhstest/uhc_based_envs/data/stretching/outputs_EmbodiedPose/wild_processed.pkl")
    parser.add_argument("--mode", type=str, default="vis")
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hide_expert", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--shift_kin", action="store_false", default=True)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    parser.add_argument("--filter_res", action="store_true", default=False)
    parser.add_argument("--no_filter_2d", action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    # Flags
    flags.debug = args.debug
    flags.no_filter_2d = args.no_filter_2d
    cfg.no_log = True
    if args.no_fail_safe:
        cfg.fail_safe = False
        
    return cfg, args


if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))
    custom_cfg, custom_args = get_config()
    
    custom_maker = Custom_HumanoidKinEnvRes(cfg=custom_cfg, args=custom_args)
    custom_env = custom_maker(cfg=custom_cfg)
    
    print(f"custom_env.action_space : {custom_env.action_space}")
    print(f"custom_env.observation_space : {custom_env.observation_space}")
    
    dataset_path = '/home/nhstest/uhc_based_envs/data/stretching/sa_seq.pkl'
    with open(dataset_path, 'rb') as pkl_file:
        dataset = pickle.load(pkl_file)
    print(f"dataset path : {dataset_path}")
    print(f"dataset type : {type(dataset)}")
    
    initial_state = dataset['states'][0]
    
    
    custom_env.set_mode('test')
    frames = []
    for action in dataset['actions']:
        custom_env.step(action)
        frames.append(custom_maker.get_sim_render(custom_env))
    
    custom_maker.save_video(frames=frames)
        
    
    
    