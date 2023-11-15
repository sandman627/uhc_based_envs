import os
import sys
import argparse

from typing import List

import pickle
import cv2

import numpy as np
import torch

from uhc.utils.flags import flags
from uhc.khrylib.utils.torch import to_device

from embodiedpose.agents import agent_dict
from embodiedpose.data_loaders import data_dict

from embodiedpose.utils.video_pose_config import Config
from embodiedpose.envs.humanoid_kin_res import HumanoidKinEnvRes
from embodiedpose.models.kin_policy_humor_res import KinPolicyHumorRes




class Custom_HumanoidKinEnvRes():
    def __init__(self) -> None:
        pass



    def test_HumanoidKinEnvRes(self):
        
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

        # thread for 'vis'
        if cfg.mode == "vis":
            cfg.num_threads = 1
            
        # Humanoid model
        if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
            cfg.robot_cfg["model"] = "smplx"

        cfg.data_specs["train_files_path"] = [(cfg.data,"scene_pose")]
        cfg.data_specs["test_files_path"] = [(cfg.data,"scene_pose")]

        # dtype
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        
        # mode & global_start_fr
        mode = 'test'
        global_start_fr = 0
        
        # device & seed
        device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_index)
        print(f"Using: {device}")
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # data_loader
        data_loader = self.get_data_loader(cfg=cfg)

        # policy_net
        policy_net = self.get_policy_net(
            cfg=cfg,
            data_loader=data_loader,
            device=device,
            dtype=dtype,
            global_start_fr=global_start_fr,
            mode=mode
        )

        # Get Env
        my_env = self.get_env(cfg, data_loader, policy_net, global_start_fr)
        print(f"action space : {my_env.action_space}")
        print(f"observation space : {my_env.observation_space}")
        print(f"action dim : {my_env.action_dim}")
        print(f"observation dim : {my_env.obs_dim}")
        '''
        action space: 315 but when evaluating, the action space is (114,)
        why is it different? : 
        '''
        
        # Get the camera image
        width, height = 640, 480  # Specify the desired width and height
        resolution = [width, height]
        camera_name = None  # If None, the default camera is used
        if camera_name is None:
            camera_id = -1
        else:
            camera_id = my_env.sim.model.camera_name2id(camera_name)
        
        # image of simulation
        my_img = np.flipud(my_env.sim.render(width=width, height=height))
        cv2.imwrite("/home/nhstest/uhc_based_envs/src/temp/saved_image.png", my_img)
        
        return my_env



    def get_env(self, cfg, data_loader, policy_net, global_start_fr=0):
        """load CC model"""
        with torch.no_grad():
            data_sample = data_loader.sample_seq(fr_num=20, fr_start=global_start_fr)
            context_sample = policy_net.init_context(data_sample, random_cam=True)
        
        '''
        context_sample: linked to 'init_state', seems like initial state or pose
        '''
        env = HumanoidKinEnvRes(
            cfg, 
            init_context=context_sample, 
            cc_iter=cfg.policy_specs.get('cc_iter', -1), 
            mode="train", 
            agent=None
        )
        env.seed(cfg.seed)
        return env


    def get_data_loader(self, cfg):
        train_files_path = cfg.data_specs.get("train_files_path", [])
        test_files_path = cfg.data_specs.get("test_files_path", [])
        train_data_loaders, test_data_loaders, data_loader = [], [], None

        if len(train_files_path) > 0:
            for train_file, dataset_name in train_files_path:

                data_loader = data_dict[dataset_name](cfg, [train_file], multiproess=False)
                train_data_loaders.append(data_loader)

        if len(test_files_path) > 0:
            for test_file, dataset_name in test_files_path:
                data_loader = data_dict[dataset_name](cfg, [test_file], multiproess=False)
                test_data_loaders.append(data_loader)

        data_loader = np.random.choice(train_data_loaders)
        return data_loader


    def get_policy_net(self, cfg, data_loader, device, dtype, global_start_fr, mode):
        data_sample = data_loader.sample_seq(fr_num=20, fr_start=global_start_fr)
        policy_net = KinPolicyHumorRes(
            cfg, 
            data_sample, 
            device=device, 
            dtype=dtype, 
            mode=mode, 
            agent=None
        )
        to_device(device, policy_net)
        return policy_net




    def save_video(self, frames:List, video_path="/home/nhstest/uhc_based_envs/src/temp/test.mp4"):
        print("video recording..")
        height, width, layers  = frames[0].shape
        size = (width,height)
        fps = 15
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), size)
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()




if __name__ == "__main__":
    print("Testing : ", os.path.basename(__file__))
    test_HumanoidKinEnvRes()