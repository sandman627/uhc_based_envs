from typing import List
import os
import time
import argparse
import pickle
import joblib
import yaml
from tqdm import tqdm

import imageio
import cv2

import numpy as np

import mujoco_py
from mujoco_py import GlfwContext
from mujoco_py import load_model_from_path, load_model_from_xml, MjSim, MjViewer

from uhc.utils.config_utils.copycat_config import Config
from uhc.smpllib.smpl_robot import Robot
from uhc.khrylib.rl.envs.common.mujoco_env import MujocoEnv
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.data_process.process_smpl_data import smpl_2_entry

"""
to Run UHC outside the repo, you need some files
    config: config file
    data: with smpl model datas
    assets: from uhc repo
config file and data file are inside UHC repo, but assets must be downloaded from SMPL website

"""


def read_config(config_filename:str="copycat_5"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=config_filename)
    args = parser.parse_args()
    print("Checking Config File : ", args.cfg)
    
    print("Parsed Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    
    print("config Parsed Arguments:")
    for cf in vars(cfg):
        print(f"{cf}: {getattr(cfg, cf)}")


def running_env():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="copycat_40")
    args = parser.parse_args()
    
    print("Checking Config File : ", args.cfg)

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    # cfg = yaml.safe_load(open("/home/nhstest/uhc_based_envs/src/config/copycat_40.yml",'r'))
    
    
    cfg.robot_cfg["model"] = "smplx"
    cfg.robot_cfg["mesh"] = True
    smpl_robot = Robot(cfg.robot_cfg, masterfoot=False)
    params_names = smpl_robot.get_params(get_name=True)

    # print("Checking robot_cfg : ", cfg.robot_cfg)


    smpl_robot.load_from_skeleton(gender=[1])
    smpl_robot.write_xml(f"test.xml")
    model = load_model_from_path(f"test.xml")

    print(f"mass {mujoco_py.functions.mj_getTotalmass(model)}")
    sim = MjSim(model)
    t1 = time.time()

    # viewer = MjViewer(sim)
    # viewer = mujoco_py.MjRenderContextOffscreen(sim)
    print(sim.data.qpos.shape, sim.data.ctrl.shape)

    # Get the camera image
    width, height = 640, 480  # Specify the desired width and height
    resolution = [width, height]
    camera_name = None  # If None, the default camera is used
    if camera_name is None:
        camera_id = -1
    else:
        camera_id = sim.model.camera_name2id(camera_name)
        
    frames = []
    
    print("sim : ", sim)
    # print("viewer.sim : ", viewer.sim)
    # set Vars for recording
    # viewer._record_video = True
    # viewer._video_path = "/home/nhstest/uhc_based_envs/src/temp/video_%07d.mp4"
    ###
    
    # Get sim vals
    print("-------------------------------------------")
    print(f"Sim Attributes : ")
    attributes = {attr: getattr(sim, attr) for attr in dir(sim) if not attr.startswith('__')}
    for key, val in attributes.items():
        print(f"{key} : {val}")
        
    print(f"type of sim.data : {type(sim.data)}")
    # exit()    
    

    video_length = 300
    stop = False
    paused = False
    sim.data.qpos[2] = 1
    sim.forward()
    timestep = 0
    while True:
        # sim.data.qpos[2] = 1
        # sim.data.qpos[7 + 42] = -np.pi/6
        # sim.data.qpos[7 + 44] = -np.pi/2

        # sim.data.qpos[7 + 59] = np.pi/2
        # sim.data.qpos[7 + 57] = -np.pi/6
        # sim.data.qpos[7 + 14] = -np.pi/3
        # sim.data.qpos[7 + 28] = -np.pi/4
        # sim.data.qpos[7 + 31] = -np.pi/4
        sim.data.ctrl[:] = 0
        # sim.forward()
        sim.step()
        frames.append(np.flipud(sim.render(width=width, height=height)))
        
        # viewer.render()
        
        
        timestep += 1
        if timestep >= video_length:
            break
    
    
    save_video(frames=frames)
    


def checking_HumanoidEnv():
    data_dir = "/hdd/zen/dev/copycat/MST10192 Final Working/sample.pkl"
    # data_dir = '/hdd/zen/data/video_pose/prox/results/sample.pkl'
    # data_dir = "sample_data/h36m_test_30_fitted_grad.pkl"
    # data_dir = "sample_data/h36m_train_30_fitted_grad_test.pkl"
    # data_dir = "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_full.p"
    # data_dir = "sample_data/h36m_train_30_fitted_grad.pkl"
    # data_dir = "sample_data/egopose_mocap_smpl_grad.pkl"
    # data_dir = "sample_data/h36m_all_smpl.pkl"
    # data_dir = "sample_data/relive_mocap_smpl_grad.pkl"
    # data_dir = "sample_data/relive_wild_smpl.pkl"
    # data_dir = "sample_data/relive_ar_smpl.pkl"
    # data_dir = "sample_data/relive_third_smpl.pkl"
    # data_dir = "/hdd/zen/data/copycat/seqs/AIST++/aist_smpl.pkl"
    # fix_feet = False
    fix_feet = True
    data_res = {}
    seq_length = -1
    cfg = Config(cfg_id="copycat_5", create_dirs=False)

    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    random_expert = data_loader.sample_seq()
    env = HumanoidEnv(
        cfg, init_expert=random_expert, data_specs=cfg.data_specs, mode="test"
    )
    
    print(f"env.action_dim : {env.action_dim}")
    print(f"env.obs_dim : {env.obs_dim}")
    exit()

    # target_frs = [20,30,40] # target framerate
    video_annot = {}
    counter = 0
    seq_counter = 0
    # gnd_threh = -0.15
    # feet_offset = -0.015
    # begin_feet_thresh = 0.3
    gnd_threh = -1
    feet_offset = -0.015
    begin_feet_thresh = 50

    # model_file = f'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    data_db = joblib.load(data_dir)
    all_data = list(data_db.items())
    np.random.shuffle(all_data)
    pbar = tqdm(all_data)
    for (k, v) in pbar:
        pbar.set_description(k)
        entry = smpl_2_entry(
            env=env,
            seq_name=k,
            smpl_dict=v,
            gnd_threh=gnd_threh,
            feet_offset=feet_offset,
            begin_feet_thresh=begin_feet_thresh,
            fix_feet=fix_feet,
        )
        if not entry is None:
            data_res[k] = entry
            counter += 1
        # if counter > 10:
        # break

    # output_file_name = "sample_data/h36m_all_qpos.pkl"
    # output_file_name = "sample_data/relive_mocap_qpos_grad.pkl"
    # output_file_name = "sample_data/relive_wild_qpos.pkl"
    # output_file_name = "sample_data/relive_ar_qpos.pkl"
    # output_file_name = "sample_data/relive_third_qpos.pkl"
    # output_file_name = "/hdd/zen/data/copycat/seqs/AIST++/aist_qpos.pkl"
    # output_file_name = "sample_data/egopose_mocap_qpos_grad.pkl"
    # output_file_name = "sample_data/h36m_train_30_qpos.pkl"
    # output_file_name = "sample_data/h36m_test_30_qpos.pkl"
    # output_file_name = "sample_data/h36m_train_30_qpos_test.pkl"
    # output_file_name = "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_qpos_full.p"
    # output_file_name = "sample_data/prox_sample.pkl"
    output_file_name = "sample_data/dais_sample.pkl"
    print(output_file_name, len(data_res))
    joblib.dump(data_res, open(output_file_name, "wb"))


def save_video(frames:List, video_path="/home/nhstest/uhc_based_envs/src/temp/test.mp4"):
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
    # read_config()
    running_env()
    # checking_HumanoidEnv()