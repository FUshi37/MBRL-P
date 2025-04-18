import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import sys
import math
import random

from gym import GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
# 在环境初始化前设置
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16,garbage_collection_threshold:0.8"
import json

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from gym import GYM_ROOT_DIR
from gym.envs.base_task import BaseTask
from gym.utils.terrain import Terrain
from gym.utils.terrain_rec import HTerrain
from gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from gym.utils.helpers import class_to_dict
# from .hexapod_robot_config import HexapodRobotCfg
from gym.envs.hexapodMBRL_config import MBRLHexapodCfg
from rsl_rl.datasets.motion_loader import AMPLoader
import cv2
from gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
import torch
import gc

def print_memory_usage():
    total_memory = 0
    tensor_list = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    mem_size = obj.element_size() * obj.nelement() / 1024 ** 2  # MB
                    total_memory += mem_size
                    tensor_list.append((type(obj), obj.size(), mem_size))

        except Exception as e:
            pass  # 防止一些非张量对象报错

    # 按照显存占用从大到小排序
    tensor_list.sort(key=lambda x: x[2], reverse=True)

    print(f"Total CUDA memory allocated: {total_memory:.2f} MB")
    for t in tensor_list:
        print(f"Tensor Type: {t[0]}, Size: {t[1]}, Memory: {t[2]:.2f} MB")

# no used
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET

def memory_monitor(func):
    def wrapper(*args, **kwargs):
        print(f"\n=== 进入 {func.__name__} 前 ===")
        print(torch.cuda.memory_summary(abbreviated=True))
        
        result = func(*args, **kwargs)
        
        print(f"\n=== 离开 {func.__name__} 后 ===")
        print(torch.cuda.memory_summary(abbreviated=True))
        return result
    return wrapper

def quat_to_euler(quat):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
    """
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4) + torch.pi / 2
    yaw_z = torch.where(yaw_z > 2 * torch.pi, yaw_z - 2 * torch.pi, yaw_z)
    
    return roll_x, pitch_y, yaw_z # in radians

class HexapodRobot(BaseTask): 
    def __init__(self, cfg: MBRLHexapodCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # print(f"\n=== 进入 step 前 ===")
        #         # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        self.cfg = cfg
        
        self.sim_params = sim_params
        self.height_samples = None
        # for debug
        self.debug_viz = True
        self.lookat_id = 0
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[0], self.cfg.depth.resized[1]),
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        self.velocity_ema_list = []
        self.EMA_LEN = 25
        
        if self.cfg.terrain.is_plane:
            self.cfg.env.height_dim = 0
            self.cfg.depth.use_camera = False
        if self.cfg.depth.use_camera:
            self.cfg.depth.update_interval = 10
            self.cfg.terrain.num_cols = 2
            self.cfg.terrain.terrain_length = 2.
            self.cfg.terrain.terrain_width = 2.
        
        # if not self.headless:
        #     self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)   
         # 初始化视频录制器（只在第一个环境录制） 
        if self.headless:
            pass
        else:
            self.record_video = False
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)   
        self.start_record = False
        
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        
        self.global_counter = 0
        self.total_env_steps_counter = 0

        self.latency_range = [int((self.cfg.domain_rand.latency_range[0] + 1e-8) / self.sim_params.dt),
                                 int((self.cfg.domain_rand.latency_range[1] - 1e-8) / self.sim_params.dt) + 1]

        if self.cfg.rewards.reward_curriculum:
            self.reward_curriculum_coef = [schedule[2] for schedule in self.cfg.rewards.reward_curriculum_schedule]

        # For Hexapod AMP set False
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)
         
        with open("./motorcommand.txt", "r") as f:    
            self.motorcommand = json.load(f)
            
        # # 设置参数 正弦测试
        # amplitude = 0.5           # 振幅
        # frequency = 2*np.pi           # 频率 (Hz)
        # sampling_rate = 1000    # 采样率 (samples per second)
        # duration = 1            # 持续时间 (秒)
        # # 生成时间序列
        # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        # # 生成正弦信号，并确保它是 [1, 18] 形状的列表
        # signal = [[amplitude * np.sin(2 * np.pi * frequency * ti) for _ in range(18)] for ti in t]
        # self.motorcommand = signal
        
        self.motorcommand_index = -1
        
        # print("\n==== 初始化后显存 ====")
        # print(f"\n=== 进入 init 后 ===")
        # # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    #@memory_monitor
    def reset(self):
        """ Reset all robots"""
        # print(f"\n=== 进入 step 前 ===")
        #         # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        torch.cuda.empty_cache()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # print(f"\n=== 进入 reset 后 ===")
        #         # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        return obs, privileged_obs
    
    #@memory_monitor
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print(f"\n=== 进入 step 前 ===")
        #         # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        self.global_counter += 1
        self.total_env_steps_counter += 1

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print(torch.max(torch.abs(self.actions), dim=1))

        #for action latency
        rng = self.latency_range
        action_latency = random.randint(rng[0], rng[1])
        # print("actions: ", self.actions)
        
        # # 测试
        # if self.motorcommand_index == len(self.motorcommand) - 1:
        #     self.motorcommand_index = -1
        # if self.motorcommand_index < len(self.motorcommand) - 1:
        #     self.motorcommand_index = self.motorcommand_index + 1
        # self.actions = self.motorcommand[self.motorcommand_index]
        # # print("actions len: ", len(self.actions))
        # # self.actions[3:18] = [0] * 15 
        # # self.actions[0:18] = [0] * 18
        # # print("actions: ", self.actions)
        # self.actions = torch.tensor(self.actions, dtype=torch.float32).to(device=self.device)
        # # print(actions)
        # self.actions.view(1, 18)
        # self.actions = self.actions.repeat(2, 1)
        
        # # print("actions shape: ", self.actions.shape)

        # CPG
        
        # step physics and render each frame
        self.render()
            
        for _ in range(self.cfg.control.decimation):
            if (self.cfg.domain_rand.randomize_action_latency and _ < action_latency):
                self.torques = self._compute_torques(self.last_actions).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            if(self.cfg.domain_rand.randomize_motor_strength):
                rng = self.cfg.domain_rand.motor_strength_range
                self.torques = self.torques * torch_rand_float(rng[0], rng[1], self.torques.shape, device=self.device)


            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # For Hexapod AMP set False
        reset_env_ids = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None
        # print("commands: ", self.commands)
        # 在extras中添加监控指标
        self.extras["metrics/negative_force_ratio"] = torch.mean((self.sensor_forces[:, :, 2] < 0).float())
        self.extras["metrics/foot_velocity_z"] = torch.mean(self.rigid_body_lin_vel[:, self.feet_indices, 2].abs())
        # print(f"\n=== 进入 step 后 ===")
        # # 打印当前分配情况
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        # # 获取详细统计
        # print(torch.cuda.memory_stats())

        # # 关键指标：
        # print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # print(f"最大缓存内存: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
        # print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image
    
    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]
    
    #@memory_monitor
    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        start_time = time()
        self.gym.start_access_image_tensors(self.sim)
        for i in range(self.num_envs):
        # for i in range(len(self.depth_index)):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                # self.envs[self.depth_index[i]],
                                                                self.envs[i],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)
        self.gym.end_access_image_tensors(self.sim)
        # print('acquiring depth image time:', time()-start_time)

    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
            # print("policy obs: ", policy_obs.shape)
        return policy_obs
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.prev_base_lin_vel[:] = self.base_lin_vel[:].clone()

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # the original code call _post_physics_step_callback before compute reward, which seems unreasonable. e.g., the
        # current action follows the current commands, while _post_physics_step_callback may resample command, resulting a low reward.
        self._post_physics_step_callback()
        
        current_vel = self.base_lin_vel[:, :3]  # 获取当前线速度
        self.velocity_ema_buffer = torch.cat([
            self.velocity_ema_buffer[:, 1:, :], 
            current_vel.unsqueeze(1)
        ], dim=1)
        self.velocity_ema = torch.mean(self.velocity_ema_buffer, dim=1)
            
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # For Hexapod AMP set False
        # terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)

        self.update_depth_buffer()

        # after reset idx, the base_lin_vel, base_ang_vel, projected_gravity, height has changed, so should be re-computed
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # self._post_physics_step_callback()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            if self.cfg.depth.use_camera and True:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            # # 在render循环中添加力矢量绘制
            # force = self.sensor_forces[:, :, 2].cpu().numpy()
            # start_point = self.rigid_body_pos[:, self.feet_indices, :].cpu().numpy()
            # end_point = start_point + force * 0.01  # 缩放力矢量
            # color = (1, 0, 0) if force[2] < 0 else (0, 1, 0)  # 红色表示负向力
            # self.gym.add_lines(self.viewer, self.envs[:], 1,
            #                 [start_point[0], start_point[1], start_point[2],
            #                 end_point[0], end_point[1], end_point[2]],
            #                 color)

        return env_ids
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        
        # 选出 terrain_levels < 3 的环境
        valid_envs = self.terrain_levels < 3  # shape: (num_envs,)

        # 只有 valid_envs 中的环境才检查 termination_contact_indices 的接触力
        contact_forces_norm = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        contact_violation = torch.any(contact_forces_norm > 1.0, dim=1)

        # 只对 terrain_levels < 3 的环境应用该条件
        self.reset_buf = torch.where(valid_envs, contact_violation, torch.zeros_like(contact_violation))

        vel_error = self.base_lin_vel[:, 1] - self.commands[:, 1]
        self.vel_violate = ((vel_error > 1.5) & (self.commands[:, 0] < 0.)) | ((vel_error < -1.5) & (self.commands[:, 0] > 0.))
        # self.vel_violate = ((vel_error > 0.03) & (self.commands[:, 1] < 0.)) | ((vel_error < -0.03) & (self.commands[:, 1] > 0.))
        if self.cfg.terrain.curriculum:
            self.vel_violate *= (self.terrain_levels > 3) # terrain_levels ???
        
        # # 限制高度
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # self.base_height = base_height < 0.55
        # self.reset_buf |= self.base_height

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # print("self.max_episode_length: ", self.max_episode_length)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.vel_violate

        self.fall = (self.root_states[:, 9] < -3.) | (self.projected_gravity[:, 2] > 0.)
        # print("reset vel violate: ", self.vel_violate)
        # print("reset timeout buf: ", self.time_out_buf)
        # print("reset fall: ", self.fall)
        self.reset_buf |= self.fall
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # reset robot states // For Hexapod AMP set False
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)


        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0
        self.latency_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.no_feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.velocity_ema_buffer[env_ids] = 0  # 完全重置
        self.velocity_ema[env_ids] = 0
        self.smooth_air_time = torch.zeros_like(self.feet_air_time, device=self.device)
        self.feet_contact_time[env_ids] = 0.
        mask = self.stability_counter[env_ids] >= self.cfg.terrain.curriculum_counter
        self.stability_counter[env_ids[mask]] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

            self.extras["episode"]["max_command_flat_x"] = self.command_ranges["flat_lin_vel_x"][1]
            self.extras["episode"]["max_command_flat_yaw"] = self.command_ranges["flat_ang_vel_yaw"][1]

            self.extras["episode"]["push_interval_s"] = self.cfg.domain_rand.push_interval_s
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # reward curriculum
            if self.cfg.rewards.reward_curriculum:
                for j in range(len(self.cfg.rewards.reward_curriculum_term)):
                    if(name == self.cfg.rewards.reward_curriculum_term[j]):
                        rew *= self.reward_curriculum_coef[j]
                # print('reward:', name, ' coef:', self.reward_curriculum_coef)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


    def compute_observations(self):
        """ Computes observations
        """
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3
                                    self.projected_gravity, # 3
                                    # self.commands[:, :3] * self.commands_scale, # 3
                                    self.commands[:, :3],
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 18
                                    self.dof_vel * self.obs_scales.dof_vel, # 18
                                    self.actions # 18
                                    ),dim=-1)
        # print("privileged_obs_buf: ", self.privileged_obs_buf.shape)
        # print("self.command: ", self.commands)
        # print("privileged_obs_command: ", self.privileged_obs_buf[:, 9:12])
        if (self.cfg.env.privileged_obs):
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights and not self.cfg.terrain.is_plane: # 187
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                # print("height: ", heights)
                # print("measured_heights: ", self.measured_heights[0])
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
                # print("privileged_obs_command: ", self.privileged_obs_buf[:, 9:12])
            # print("privileged_obs_buf measure height: ", self.privileged_obs_buf.shape)
            if self.cfg.domain_rand.randomize_friction: # 1
                self.privileged_obs_buf= torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)
                # print("randomized_frictions: ", self.randomized_frictions.shape)
            # print("privileged_obs_buf randomize friction: ", self.privileged_obs_buf.shape)
            if self.cfg.domain_rand.randomize_restitution: # 1
                self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)
                # print("randomized_restitutions: ", self.randomized_restitutions.shape)
            # print("privileged_obs_buf randomize restitution: ", self.privileged_obs_buf.shape)
            if (self.cfg.domain_rand.randomize_base_mass): # 1
                self.privileged_obs_buf = torch.cat((self.randomized_added_masses ,self.privileged_obs_buf), dim=-1)
                # print("randomized_added_masses: ", self.randomized_added_masses)
            # print("privileged_obs_buf randomize base mass: ", self.privileged_obs_buf.shape)
            if (self.cfg.domain_rand.randomize_com_pos): # 3
                self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos ,self.privileged_obs_buf), dim=-1)
            # print("privileged_obs_buf randomize com pos: ", self.privileged_obs_buf.shape)
            if (self.cfg.domain_rand.randomize_gains): # 36
                self.privileged_obs_buf = torch.cat(((self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains ,self.privileged_obs_buf), dim=-1)
                self.privileged_obs_buf = torch.cat(((self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains, self.privileged_obs_buf),
                                                    dim=-1)
                # print("randomized_p_gains: ", self.randomized_p_gains.shape)
                # print("randomized_d_gains: ", self.randomized_d_gains.shape)
            # print("privileged_obs_buf randomize gains: ", self.privileged_obs_buf.shape)
            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1) # 18
            # print("privileged_obs_buf contact force: ", contact_force.shape)
            # print("privileged_obs_buf contact force: ", self.privileged_obs_buf.shape)
            contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1 # 12
            self.privileged_obs_buf = torch.cat((contact_flag, self.privileged_obs_buf), dim=-1)
            # print("privileged_obs_buf contact flag: ", contact_flag.shape)
            # print("privileged_obs_buf contact flag: ", self.privileged_obs_buf.shape)
        # print("privileged_obs_command: ", self.privileged_obs_buf[:, 9:12])
        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec
        # print("privileged_obs_buf add noise: ", self.privileged_obs_buf.shape)
        
        # # 将所有的观测数据存储到文件中
        # with open("privileged_obs_buf.txt", "a") as f:
        #     # for i in range(self.privileged_obs_buf.shape[0]):
        #     for j in range(self.privileged_obs_buf.shape[1]):
        #         f.write(str(self.privileged_obs_buf[0][j].item()) + " ")
        #     f.write("\n")
        
        
        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        elif self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            # self.terrain = HTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
            # if self.cfg.terrain.curriculum:
            #     pass
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        # self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            # if env_id==0:
            #     # prepare friction randomization
            #     friction_range = self.cfg.domain_rand.friction_range
            #     num_buckets = 64
            #     bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            #     friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
            #     self.friction_coeffs = friction_buckets[bucket_ids]
            #
            # for s in range(len(props)):
            #     props[s].friction = self.friction_coeffs[env_id]
            rng = self.cfg.domain_rand.friction_range
            self.randomized_frictions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].friction = self.randomized_frictions[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            rng = self.cfg.domain_rand.restitution_range
            self.randomized_restitutions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].restitution = self.randomized_restitutions[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            self.randomized_added_masses[env_id] = added_mass
            props[0].mass += added_mass

        # randomize com position
        if self.cfg.domain_rand.randomize_com_pos:
            rng = self.cfg.domain_rand.com_x_pos_range
            com_x_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,0] = com_x_pos
            rng = self.cfg.domain_rand.com_y_pos_range
            com_y_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,1] = com_y_pos
            rng = self.cfg.domain_rand.com_z_pos_range
            com_z_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,2] = com_z_pos
            props[0].com +=  gymapi.Vec3(com_x_pos,com_y_pos,com_z_pos)

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            # heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
            # self.commands[:self.roughflat_start_idx, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:self.roughflat_start_idx, 3] - heading), -1., 1.)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        else:
            self.commands[:, 2] = torch.atan2(self.commands[:, 1], self.commands[:, 0])
            
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.measured_forward_heights = self._get_forward_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch.atan2(self.commands[env_ids, 1], self.commands[env_ids, 0])

        #resample commands for rough flat terrain
        # flat_env_ids = env_ids[torch.where(env_ids >= self.roughflat_start_idx)]
        # if(len(flat_env_ids) > 0):
        #     self.commands[flat_env_ids, 0] = torch_rand_float(self.command_ranges["flat_lin_vel_x"][0],
        #                                                  self.command_ranges["flat_lin_vel_x"][1], (len(flat_env_ids), 1),
        #                                                  device=self.device).squeeze(1)
        #     self.commands[flat_env_ids, 1] = torch_rand_float(self.command_ranges["flat_lin_vel_y"][0],
        #                                                  self.command_ranges["flat_lin_vel_y"][1], (len(flat_env_ids), 1),
        #                                                  device=self.device).squeeze(1)
        #     self.commands[flat_env_ids, 2] = torch_rand_float(self.command_ranges["flat_ang_vel_yaw"][0],
        #                                                  self.command_ranges["flat_ang_vel_yaw"][1], (len(flat_env_ids), 1),
        #                                                  device=self.device).squeeze(1)

        # # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # # set heading command for tilt envs to zero
        # self.commands[self.tilt_start_idx:self.tilt_end_idx, 3] = 0
        # self.commands[self.pit_start_idx:self.pit_end_idx, 3] = 0
        # # self.commands[self.gap_start_idx:self.gap_end_idx, 3] = 0
            
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        # print("actions_scaled: ", actions_scaled)
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            # mask = distance > 1.6  # 创建布尔掩码
            # filtered_env_ids = env_ids[mask]  # 筛选出对应的 env_ids

            # # 打印符合条件的 env_ids 和对应的 distance
            # print("env_ids with distance > 1.6:", filtered_env_ids)
            # print("corresponding distances:", distance[mask])
            # self.root_states[env_ids, :2] += torch_rand_float(-0.01, 0.01, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # First ground plane
        # # the base y position of tilt and gap envs can not deviate too far from the origin center
        # tilt_env_ids = env_ids[torch.where(env_ids >= self.tilt_start_idx)]
        # tilt_env_ids = tilt_env_ids[torch.where(tilt_env_ids < self.tilt_end_idx)]
        # gap_env_ids = env_ids[torch.where(env_ids >= self.gap_start_idx)]
        # gap_env_ids = gap_env_ids[torch.where(gap_env_ids < self.gap_end_idx)]
        # tilt_and_gap_env_ids = torch.concatenate((tilt_env_ids, gap_env_ids))

        # if self.custom_origins:
        #     self.root_states[tilt_and_gap_env_ids] = self.base_init_state
        #     self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]
        #     self.root_states[tilt_and_gap_env_ids, :1] += torch_rand_float(-1., 1., (len(tilt_and_gap_env_ids), 1), device=self.device) # x position within 1m of the center
        #     self.root_states[tilt_and_gap_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(tilt_and_gap_env_ids), 1),
        #                                                        device=self.device)
        # else:
        #     self.root_states[tilt_and_gap_env_ids] = self.base_init_state
        #     self.root_states[tilt_and_gap_env_ids, :3] += self.env_origins[tilt_and_gap_env_ids]

        # the base y position of gap env can not deviate too far from the origin center
        # gap_env_ids = env_ids[torch.where(env_ids >= self.gap_start_idx)]
        # gap_env_ids = gap_env_ids[torch.where(gap_env_ids < self.gap_end_idx)]
        # if self.custom_origins:
        #     self.root_states[gap_env_ids] = self.base_init_state
        #     self.root_states[gap_env_ids, :3] += self.env_origins[gap_env_ids]
        #     self.root_states[gap_env_ids, :1] += torch_rand_float(-1., 1., (len(gap_env_ids), 1), device=self.device) # x position within 1m of the center
        #     self.root_states[gap_env_ids, 1:2] += torch_rand_float(-0.0, 0.0, (len(gap_env_ids), 1),
        #                                                        device=self.device)
        # else:
        #     self.root_states[gap_env_ids] = self.base_init_state
        #     self.root_states[gap_env_ids, :3] += self.env_origins[gap_env_ids]

        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.02, 0.02, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel #0.5
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    # def update_reward_curriculum(self, current_iter):
    #     for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
    #         percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
    #                      (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
    #         percentage = max(min(percentage, 1), 0)
    #         self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
    #                                       percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        command_vel = self.commands[env_ids, :2]
        # distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1).to(self.device)
        distance_y = self.root_states[env_ids, 1] - self.env_origins[env_ids, 1]
        distance_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        target_yaw = torch.atan2(command_vel[:, 1], command_vel[:, 0])
        current_yaw = quat_to_euler(self.base_quat)[2][env_ids]
        yaw_diff = torch.abs(target_yaw - current_yaw)
        MOVE_UP_CONDITIONS = {
            'x_error': 0.05,      # x方向允许10%的相对误差或20cm绝对误差
            'y_error': (torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.85) - self.terrain_levels[env_ids]/30,      # y方向允许15%的相对误差
            'yaw_error': 0.4      # 偏航角允许10%的误差
        }
        
        MOVE_DOWN_CONDITIONS = {
            'x_error': 0.5,       # x方向超50%相对误差或50cm绝对误差
            'y_error': (torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5),       # y方向超过50%相对误差
            'yaw_error': 1.5      # 偏航角超过50%误差
        }
        move_up = (
            (distance_x < MOVE_UP_CONDITIONS['x_error']) &
            (distance_y > MOVE_UP_CONDITIONS['y_error']) &
            (yaw_diff < MOVE_UP_CONDITIONS['yaw_error'])
        )
        
        move_down = (
            (distance_x > MOVE_DOWN_CONDITIONS['x_error']) |
            (distance_y < MOVE_DOWN_CONDITIONS['y_error']) |
            (yaw_diff > MOVE_DOWN_CONDITIONS['yaw_error'])
        )
        # # robots that walked far enough progress to harder terains
        # move_up = (distance > torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.9)#self.terrain.env_length / 2
        # # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) #* ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        
        if True:
            self.terrain_levels = torch.ones((self.num_envs,), device=self.device).long() * 3
            # 制定terrain_levels为2
            # self.terrain_levels = torch.ones((self.num_envs,), device=self.device).long() * 7
            
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _update_terrain_curriculum_vel(self, env_ids):
        """ Implements the game-inspired curriculum with improved logic for lateral movement and rotation.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            return

        # 获取命令速度和实际速度
        command_vel = self.commands[env_ids, :2]
        actual_vel = self.velocity_ema[env_ids, :2]
        command_x = command_vel[:, 0]
        command_y = command_vel[:, 1]
        actual_x = actual_vel[:, 0]
        actual_y = actual_vel[:, 1]

        # 自适应误差计算阈值
        VELOCITY_EPSILON = 0.01  # 1 cm/s 作为零速度阈值

        # 改进的速度误差计算（针对x=0的特殊情况）
        x_error = torch.where(
            torch.abs(command_x) > VELOCITY_EPSILON,
            torch.abs(command_x - actual_x) / (torch.abs(command_x) + 1e-6),  # 相对误差
            torch.abs(actual_x)  # 绝对误差
        )
        
        y_error = torch.where(
            torch.abs(command_y) > VELOCITY_EPSILON,
            torch.abs(command_y - actual_y) / (torch.abs(command_y) + 1e-6),  # 相对误差
            torch.abs(actual_y)  # 绝对误差
        )

        # 改进的yaw误差计算（考虑目标方向）
        target_yaw = torch.atan2(command_vel[:, 1], command_vel[:, 0])
        current_yaw = quat_to_euler(self.base_quat)[2][env_ids]
        yaw_diff = torch.abs(target_yaw - current_yaw)
        # print("yaw_diff: ", yaw_diff)
        
        # 将角度差规范化到[0, π]范围内
        yaw_diff = torch.min(yaw_diff, 2 * torch.pi - yaw_diff)
        yaw_error = yaw_diff / (torch.abs(target_yaw) + 1e-6)

        # 课程条件参数（可根据需要调整）
        MOVE_UP_CONDITIONS = {
            'x_error': 0.05,      # x方向允许10%的相对误差或10cm绝对误差
            'y_error': 0.09 + self.terrain_levels[env_ids]/200,      # y方向允许15%的相对误差
            'yaw_error': 0.065      # 偏航角允许10%的误差
        }
        
        MOVE_DOWN_CONDITIONS = {
            'x_error': 0.1,       # x方向超50%相对误差或50cm绝对误差
            'y_error': 0.85,       # y方向超过50%相对误差
            'yaw_error': 0.85      # 偏航角超过50%误差
        }

        # 改进的课程条件
        move_up = (
            (x_error < MOVE_UP_CONDITIONS['x_error']) &
            (y_error < MOVE_UP_CONDITIONS['y_error']) &
            (yaw_error < MOVE_UP_CONDITIONS['yaw_error'])
        )
        
        move_down = (
            (x_error > MOVE_DOWN_CONDITIONS['x_error']) |
            (y_error > MOVE_DOWN_CONDITIONS['y_error']) |
            (yaw_error > MOVE_DOWN_CONDITIONS['yaw_error'])
        )

        # 增加稳定性检查（防止单次波动）
        self.stability_counter[env_ids] = torch.where(
            move_up,
            torch.clamp(self.stability_counter[env_ids] + 1, min=0, max=10),
            torch.where(
                move_down,
                torch.clamp(self.stability_counter[env_ids] - 2, min=-50, max=0),
                self.stability_counter[env_ids]
            )
        )
        
        # 最终决策需要连续满足条件
        move_up = (self.stability_counter[env_ids] >= self.cfg.terrain.curriculum_counter)
        move_down = (self.stability_counter[env_ids] <= -50)

        # 执行地形难度调整
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clamp(self.terrain_levels[env_ids], 0)
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above a certain percentage of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_lin_vel"]) +
                torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_ang_vel"]) > 1.65
                ):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_forward_x_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_y_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_y_curriculum)

            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.025,
                                                          -self.cfg.commands.max_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.025, 0.,
                                                          self.cfg.commands.max_ang_vel_yaw_curriculum)

            self.command_ranges["flat_lin_vel_x"][0] = np.clip(self.command_ranges["flat_lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["flat_lin_vel_x"][1] = np.clip(self.command_ranges["flat_lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_forward_x_curriculum)
            self.command_ranges["flat_lin_vel_y"][0] = np.clip(self.command_ranges["flat_lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_y_curriculum, 0.)
            self.command_ranges["flat_lin_vel_y"][1] = np.clip(self.command_ranges["flat_lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_y_curriculum)

            self.command_ranges["flat_ang_vel_yaw"][0] = np.clip(self.command_ranges["flat_ang_vel_yaw"][0] - 0.1,
                                                          -self.cfg.commands.max_flat_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["flat_ang_vel_yaw"][1] = np.clip(self.command_ranges["flat_ang_vel_yaw"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_flat_ang_vel_yaw_curriculum)

            self.cfg.domain_rand.push_interval_s = max(self.cfg.domain_rand.push_interval_s - 0.5, self.cfg.domain_rand.min_push_interval_s)
            self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_start_dim = self.privileged_dim - 3 # last 3-dim is the linear vel
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[noise_start_dim:noise_start_dim+3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[noise_start_dim+3:noise_start_dim+6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[noise_start_dim+6:noise_start_dim+9] = noise_scales.gravity * noise_level
        noise_vec[noise_start_dim+9:noise_start_dim+12] = 0. # commands
        noise_vec[noise_start_dim+12:noise_start_dim+24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[noise_start_dim+24:noise_start_dim+36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[noise_start_dim+36:noise_start_dim+48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[noise_start_dim+48:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis


        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 6, 6)[..., :3]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.rigid_body_lin_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[...,7:10]


        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros_like(self.last_actions)
        # for latency
        self.latency_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                           requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.no_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.no_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.antidragging_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.prev_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.forward_height_points = self._init_forward_height_points()
        self.measured_heights = 0
        self.measured_forward_heights = 0
        self.velocity_ema_buffer = torch.zeros(self.num_envs, self.EMA_LEN, 3,  # 假设存储线速度的xyz三个维度
            device=self.device, requires_grad=False)
        self.velocity_ema = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            self.p_gains[i] = self.cfg.control.stiffness['joint']
            self.d_gains[i] = self.cfg.control.damping['joint']
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
        
        self.smooth_air_time = torch.zeros_like(self.feet_air_time, device=self.device)
        self.feet_contact_time = torch.zeros_like(self.feet_air_time, device=self.device)
        self.stability_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

    def compute_randomized_gains(self, num_envs):
        p_mult = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        d_mult = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        return p_mult * self.p_gains, d_mult * self.d_gains


    # def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
    #     theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
    #     l_up = 0.2
    #     l_low = 0.2
    #     l_hip = 0.08505 * l_hip_sign
    #     leg_distance = torch.sqrt(l_up**2 + l_low**2 +
    #                             2 * l_up * l_low * torch.cos(theta_knee))
    #     eff_swing = theta_hip + theta_knee / 2

    #     off_x_hip = -leg_distance * torch.sin(eff_swing)
    #     off_z_hip = -leg_distance * torch.cos(eff_swing)
    #     off_y_hip = l_hip

    #     off_x = off_x_hip
    #     off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
    #     off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
    #     return torch.stack([off_x, off_y, off_z], dim=-1)

    # def foot_positions_in_base_frame(self, foot_angles):
    #     foot_positions = torch.zeros_like(foot_angles)
    #     for i in range(4):
    #         foot_positions[:, i * 3:i * 3 + 3].copy_(
    #             self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
    #     foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
    #     return foot_positions

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        # self.terrain.vertices, self.terrain.triangles = self.terrain.get_box_mesh()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])


            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # print("body names: ", body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        hip_names = [s for s in body_names if self.cfg.asset.hip_name in s]
        # print("feet names: ", feet_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])



        # use the sensor to acquire contact force, may be more accurate
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)


        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        #for domain randomization
        self.randomized_frictions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_restitutions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_com_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.depth_index = np.arange(0, self.num_envs)
        self.depth_index_inverse = -np.ones(self.num_envs, dtype=np.int)
        for i in range(len(self.depth_index)):
            self.depth_index_inverse[self.depth_index[i]] = i
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-0., 0., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            # if(self.cfg.depth.use_camera and i in self.depth_index):
            self.attach_camera(i, env_handle, anymal_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])
            
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            # print("terrain_levels: ", self.terrain_levels)
            # print("terrain levels in get env origins: ", self.terrain_levels)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            # heights = self.measured_heights[i].cpu().numpy()
            heights = self.measured_forward_heights[i].cpu().numpy()
            # height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.forward_height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_forward_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_forward_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_forward_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_forward_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_forward_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_forward_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_forward_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_forward_height_points), self.forward_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_forward_height_points), self.forward_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_forward_map(self):
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_forward_heights, -1,
                             1.) * self.obs_scales.height_measurements
        
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_ang_xy(self):
        roll = quat_to_euler(self.base_quat)[0]
        pitch = quat_to_euler(self.base_quat)[1]
        threshold = 5 * (np.pi / 180)  # 5° 转换为弧度
        roll_penalty = torch.square(torch.clamp_min(torch.abs(roll) - threshold, 0))
        pitch_penalty = torch.square(torch.clamp_min(torch.abs(pitch) - threshold, 0))
        return roll_penalty + pitch_penalty  # 只有超出阈值才有惩罚

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print("base_height: ", torch.max(base_height))
        # print("root_height: ", self.root_states[:, 2].unsqueeze(1))
        # print("measured_heights: ", self.measured_heights)
        # print("base_height_target: ", self.cfg.rewards.base_height_target)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)
        # return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_distribution(self):
        # Penalize torques
        return torch.var(torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_pos_dif(self):
        return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        # print("dof_vel: ", torch.max(self.dof_vel[0]))
        # print("max dof vel: ", torch.max(self.dof_vel[0]))
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # print("max torques: ", torch.max(self.torques[0]))
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

        # clipping tracking reward
        # lin_vel = self.base_lin_vel[:, :2].clone()
        lin_vel = self.velocity_ema[:, :2].clone() #使用平均速度，并非ema速度
        
        # lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
        # lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
        # clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
        clip_lin_vel = lin_vel
        
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - clip_lin_vel), dim=1)
        # print("clip_lin_vel: ", clip_lin_vel[0])
        # print("vel_error: ", lin_vel_error)
        
        # 记录第一个环境的y方向速度（每次写入文件）
        y_vel = clip_lin_vel[0, 1].item()  # 假设环境索引为0
        
        # # 使用追加模式写入文件（注意文件路径权限）
        # with open("clip_lin_vel_y.txt", "a") as f:
        #     f.write(f"{y_vel}\n")  # 写入后换行
            
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_y(self):
        lin_vel_y = self.velocity_ema[:, 1:2].clone() #使用平均速度，并非ema速度
        # lin_vel_y = self.base_lin_vel[:, 1:2].clone()
        clip_lin_vel = lin_vel_y
        # print("lin_vel_y: ", lin_vel_y[0])
        # lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
        # lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
        # clip_lin_vel = torch.clip(lin_vel_y, lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error_y = torch.sum(torch.abs(self.commands[:, 1:2] - clip_lin_vel), dim=1)
        
        # # 记录第一个环境的y方向速度（每次写入文件）
        # y_vel = clip_lin_vel[0, 0].item()  # 假设环境索引为0
        # # 使用追加模式写入文件（注意文件路径权限）
        # with open("clip_lin_vel_y.txt", "a") as f:
        #     f.write(f"{y_vel}\n")  # 写入后换行
        
        # print("lin_vel_y error: ", lin_vel_error_y)
        # print("rew: ", torch.exp(-lin_vel_error_y/self.cfg.rewards.tracking_sigma)[0])
        return torch.exp(-lin_vel_error_y/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_x(self):
        # lin_vel_x = self.velocity_ema[:, 0:1].clone() #使用平均速度，并非ema速度
        lin_vel_x = self.base_lin_vel[:, 0:1].clone()
        clip_lin_vel = lin_vel_x
        # lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
        # lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
        # clip_lin_vel = torch.clip(lin_vel_x, lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error_x = torch.sum(torch.abs(self.commands[:, 0:1] - clip_lin_vel), dim=1)
        
        # 记录第一个环境的y方向速度（每次写入文件）
        y_vel = clip_lin_vel[0, 0].item()  # 假设环境索引为0
        
        # # 使用追加模式写入文件（注意文件路径权限）
        # with open("clip_lin_vel_x.txt", "a") as f:
        #     f.write(f"{y_vel}\n")  # 写入后换行
            
        return torch.exp(-lin_vel_error_x/self.cfg.rewards.tracking_sigma)

    def _reward_negative_vel_y(self):
        """ 惩罚Y轴负方向速度 """
        y_vel = self.base_lin_vel[:, 1]  # 获取Y方向速度
        negative_vel = torch.clamp(-y_vel, min=0.0)  # 仅当速度为负时取值
        # print("negative_vel: ", negative_vel)
        return torch.square(negative_vel)  # 使用平方惩罚更强调较大负速度

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        if self.cfg.commands.heading_command:
            ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
            return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        else:
            target_ang = torch.atan2(self.commands[:, 1], self.commands[:, 0])
            ang = quat_to_euler(self.base_quat)[2]
            ang_error = torch.abs(target_ang - ang)
            # print("yaw_diff:  ", torch.abs(ang_error))
            # print("comamnds: ", self.commands)
            # print("target ang: ", target_ang)
            # print("base ang: ", ang)
            return torch.exp(-(ang_error)/(self.cfg.rewards.tracking_sigma*10))
            # return 0

    def _reward_x_offset_penalty(self):
        """
        计算机器人在 x 方向的偏移惩罚。
        机器人应该沿着 y 方向前进，因此希望它的 x 位置保持稳定。
        """
        # 获取机器人当前的 x 位置
        current_x_pos = self.root_states[:, 0]  # shape: (num_envs,)

        # 获取每个环境的 x 方向原点
        x_origins = self.env_origins[:, 0]  # shape: (num_envs,)
 
        # 计算 x 方向的偏移量（绝对值）
        x_offset = torch.abs(current_x_pos - x_origins)

        return x_offset

    def _reward_smooth_velocity(self):
        velocity_change = torch.norm(self.base_lin_vel - self.prev_base_lin_vel, dim=-1)
        return velocity_change
    
    def _reward_continuous_movement(self):
        """
        让机器人持续运动，防止停滞后突然加速。
        """
        min_speed = 0.015  # 允许的最小速度
        speed = torch.norm(self.base_lin_vel, dim=-1)
        # 当速度小于min_speed时，奖励为speed/min_speed，否则为0
        reward = torch.where(speed < min_speed, speed / min_speed, torch.tensor(0.0, device=self.device))
        return reward

    def _reward_feet_air_time(self):
        with torch.no_grad():  # 禁用梯度计算
            # Reward long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            # 每次reset后最开始两个step不计算此reward
            
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.2
            # contact = self.sensor_forces[:, :, 2] > 1.2
            # print("sensor force: ", self.contact_forces[:, self.feet_indices, 2])
            # print("max sensor force: ", torch.max(self.sensor_forces[:, :, 2]))
            # print("contact: ", contact)
            self.contact_filt = torch.logical_or(contact, self.last_contacts)
            # print("contact filt: ", self.contact_filt)
            self.last_contacts = contact
            # print("feed air time: ", self.feet_air_time)
            # print("contact filt: ", self.contact_filt[0])
            first_contact = (self.feet_air_time > 0.0) * self.contact_filt
            # if (self.feet_air_time[0][0] == 0.02) | (self.feet_air_time[0][0] == 0.04) :
            #     print("feet_air_time: ", self.feet_air_time[0][0], first_contact[0][0])
            # print("first_contact: ", first_contact)
            # print("contact filt: ", self.contact_filt)
            # print("feet_air_time: ", self.feet_air_time)
            # if first_contact[0][2] == True:
            #     print("feed air time: ", self.feet_air_time[0][2])
            # # sum feet air time
            # self.feet_air_time += self.dt
            # rew_airTime = torch.sum((self.feet_air_time - 1.0) * first_contact, dim=1) # reward only on first contact with the ground
            # # print("feet_air_time: ", self.feet_air_time)
            # # print("feet_air_time: ", torch.max(self.feet_air_time))
            # # print("first_contact: ", first_contact)
            # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01 #no reward for zero command
            # # print("commands: ", torch.norm(self.commands[:, :2], dim=1) > 0.01)
            # self.feet_air_time *= ~self.contact_filt
            # # print("feet ait time: ", self.feet_air_time)
            # # print("rew_airTime: ", rew_airTime)
            # return rew_airTime
            # calculate min feet air times
            
            # 独立计算每个腿的奖励
            # print("feet_air_time: ", self.feet_air_time[0][0])
            self.feet_air_time += self.dt
            # print("feet_air_time: ", self.feet_air_time[0][0])
            # print("first_contact: ", first_contact[0][3])
            per_leg_reward = (self.feet_air_time - 0.8) * first_contact  # 调整目标时间为0.8秒
            # if first_contact[0][0] == True:
                # print("per_leg_reward: ", per_leg_reward[0])
            # print("per_leg_reward: ", per_leg_reward[0])
            min_reward = torch.min(per_leg_reward, dim=1)[0]  # 取最差腿的奖励
            mean_reward = torch.mean(per_leg_reward, dim=1)    # 或取平均奖励
            # print("min_reward: ", min_reward)
            # print("mean_reward: ", mean_reward)
            # 组合奖励（示例使用最小值+平均值）
            rew_airTime = 0.1 * mean_reward + 0.9 * min_reward
            
            # 非零命令时才奖励
            rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01
            
            # 重置已触地的腿的计时器
            self.feet_air_time *= ~self.contact_filt # 接触地面时重置
            del contact, self.contact_filt, first_contact
        
        return rew_airTime
    
    # def _reward_feet_air_time(self):
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
    #     else:
    #         points = self.rigid_body_pos[:, self.feet_indices, :]

    #         # Measure ground height under the foot
    #         points += self.terrain.cfg.border_size
    #         points = (points / self.terrain.cfg.horizontal_scale).long()
    #         px = points[:, :, 0].view(-1)
    #         py = points[:, :, 1].view(-1)
    #         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    #         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    #         heights1 = self.height_samples[px, py]
    #         heights2 = self.height_samples[px + 1, py]
    #         heights3 = self.height_samples[px, py + 1]
    #         heights = torch.min(heights1, heights2)
    #         heights = torch.min(heights, heights3)

    #         ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights - self.cfg.asset.foot_radius

    #     # 足端是否接触地面（设定一个小阈值，比如 0.02m）
    #     ground_contact = foot_heights < 0.0011
    #     # print("foot_heights: ", torch.min(foot_heights))
    #     # print("ground_contact: ", ground_contact)
        
    #     # 计算每个腿的首次触地标志
    #     # first_contact = (self.feet_air_time > 0.) * self.contact_filt
    #     first_contact = (self.feet_air_time > 0.02) * ground_contact
    #     # print("first_contact: ", first_contact)
    #     # print("sensor force: ", self.sensor_forces[:, :, 2])
    #     # 更新空中时间
    #     self.feet_air_time += self.dt
    #     # print("feet_air_time: ", self.feet_air_time[0][4])
    #     # 独立计算每个腿的奖励
    #     per_leg_reward = (self.feet_air_time - 0.8) * first_contact  # 调整目标时间为0.8秒
    #     # print("per_leg_reward: ", per_leg_reward)
    #     min_reward = torch.min(per_leg_reward, dim=1)[0]  # 取最差腿的奖励
    #     mean_reward = torch.mean(per_leg_reward, dim=1)    # 或取平均奖励
    #     # print("min_reward: ", min_reward)
    #     # print("mean_reward: ", mean_reward)
    #     # 组合奖励（示例使用最小值+平均值）
    #     rew_airTime = 0.2 * mean_reward + 0.8 * min_reward
        
    #     # 非零命令时才奖励
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01
        
    #     # 重置已触地的腿的计时器
    #     self.feet_air_time *= ~ground_contact  # 接触地面时重置
        
    #     return rew_airTime
    
    def _reward_no_feet_air_time(self):
        with torch.no_grad():
            # Penalize too long steps
            # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.2
            # print("sensor force: ", self.sensor_forces[:, :, 2])
            self.no_contact_filt = torch.logical_or(contact, self.no_last_contacts)
            self.no_last_contacts = contact
            first_contact = (self.no_feet_air_time > 0.0) * self.no_contact_filt
            # print("first_contact: ", first_contact)
            # self.no_feet_air_time += self.dt
            # # rew_airTime = torch.sum((self.feet_air_time - 1.5) * first_contact, dim=1) # reward only on first contact with the ground
            # # rew_airTime = -torch.clamp(torch.sum(self.feet_air_time - 3.0, dim=1), min=0.0)
            # rew_airTime = -torch.sum(torch.clamp_min(self.no_feet_air_time - 3.0, 0.0), dim=1)
            # # print("no_feet_air_time: ", self.no_feet_air_time)
            # # print("rew_airTime: ", rew_airTime)
            # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01 #no reward for zero command
            # rew_airTime = torch.min(rew_airTime, torch.tensor(0.0, device=self.device))
            # self.no_feet_air_time *= ~self.contact_filt
            # return rew_airTime
            # 惩罚最大time
            # 更新空中时间
            # print("no_feet_air_time: ", self.no_feet_air_time[0])
            self.no_feet_air_time += self.dt
            # 独立计算每个腿的奖励
            per_leg_reward = torch.clamp_min(self.no_feet_air_time - 2.5, 0.0)  # 调整目标时间为0.8秒
            min_reward = torch.max(per_leg_reward, dim=1)[0]  # 取最差腿的奖励
            mean_reward = torch.mean(per_leg_reward, dim=1)    # 或取平均奖励
            
            # 组合奖励（示例使用最小值+平均值）
            rew_airTime = 0.1 * mean_reward + 0.9 * min_reward
            
            # 非零命令时才奖励
            rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01
            
            # 重置已触地的腿的计时器
            self.no_feet_air_time *= ~self.no_contact_filt
            del contact, self.no_contact_filt, first_contact
        return rew_airTime

    # def _reward_no_feet_air_time(self):
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
    #     else:
    #         points = self.rigid_body_pos[:, self.feet_indices, :]

    #         # Measure ground height under the foot
    #         points += self.terrain.cfg.border_size
    #         points = (points / self.terrain.cfg.horizontal_scale).long()
    #         px = points[:, :, 0].view(-1)
    #         py = points[:, :, 1].view(-1)
    #         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    #         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    #         heights1 = self.height_samples[px, py]
    #         heights2 = self.height_samples[px + 1, py]
    #         heights3 = self.height_samples[px, py + 1]
    #         heights = torch.min(heights1, heights2)
    #         heights = torch.min(heights, heights3)

    #         ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights - self.cfg.asset.foot_radius

    #     # 足端是否接触地面（设定一个小阈值，比如 0.02m）
    #     ground_contact = foot_heights < 0.0011
    #     # print("foot_heights: ", torch.min(foot_heights))
    #     # print("ground_contact: ", ground_contact)
        
    #     # contact = self.sensor_forces[:, :, 2] > 1.2
    #     # self.contact_filt = torch.logical_or(contact, self.last_contacts)
    #     # self.last_contacts = contact
        
    #     # 计算每个腿的首次触地标志
    #     # first_contact = (self.feet_air_time > 0.) * self.contact_filt
    #     first_contact = (self.feet_air_time > 0.02) * ground_contact
        
    #     # 更新空中时间
    #     self.feet_air_time += self.dt
    #     # print("feet_air_time: ", self.feet_air_time)
    #     # 独立计算每个腿的奖励
    #     per_leg_reward = torch.clamp_min(self.feet_air_time - 3.0, 0.0)  # 调整目标时间为0.8秒
    #     min_reward = torch.max(per_leg_reward, dim=1)[0]  # 取最差腿的奖励
    #     mean_reward = torch.mean(per_leg_reward, dim=1)    # 或取平均奖励
        
    #     # 组合奖励（示例使用最小值+平均值）
    #     rew_airTime = 0.2 * mean_reward + 0.8 * min_reward
        
    #     # 非零命令时才奖励
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.01
        
    #     # 重置已触地的腿的计时器
    #     self.feet_air_time *= ~ground_contact
        
    #     return rew_airTime

    # def _reward_no_feet_air_time(self):
    #     if self.cfg.terrain.mesh_type == 'plane':
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
    #     else:
    #         points = self.rigid_body_pos[:, self.feet_indices, :]

    #         # Measure ground height under the foot
    #         points += self.terrain.cfg.border_size
    #         points = (points / self.terrain.cfg.horizontal_scale).long()
    #         px = points[:, :, 0].view(-1)
    #         py = points[:, :, 1].view(-1)
    #         px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    #         py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    #         heights1 = self.height_samples[px, py]
    #         heights2 = self.height_samples[px + 1, py]
    #         heights3 = self.height_samples[px, py + 1]
    #         heights = torch.min(heights1, heights2)
    #         heights = torch.min(heights, heights3)

    #         ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
    #         foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights - self.cfg.asset.foot_radius

    #     # 足端是否接触地面（设定一个小阈值，比如 0.02m）
    #     ground_contact = foot_heights < 0.02

    #     # 更新 feet_air_time
    #     self.feet_air_time += self.dt  # 累加时间
    #     rew_airTime = -torch.sum(torch.clamp_min(self.feet_air_time - 3.0, 0.0), dim=1)
        
    #     self.feet_air_time *= ~ground_contact  # 接触地面时重置

    #     return rew_airTime

    
    def _reward_penalize_negative_force(self):
        # 获取足端在 Z 轴方向上的力 (Fz)
        contact_forces_z = self.sensor_forces[:, :, 2]  # (num_envs, num_legs)

        # 找到 Fz < 0 的力，并计算惩罚
        negative_force_mask = contact_forces_z < 0  # 形状: (num_envs, num_legs)，True 表示 Fz 为负数
        negative_force_penalty = torch.sum(torch.abs(contact_forces_z * negative_force_mask), dim=1)

        # 可以调整系数 scale_factor 控制惩罚强度
        return negative_force_penalty


    def _reward_anti_dragging(self):
        """
        避免机器人后腿拖着走，计算每条腿的触地时间，当触地时间超出阈值后给予惩罚。
        """
        with torch.no_grad():
            # 获取足部接触信息
            # contact = self.sensor_forces[:, :, 2] > 1.0  
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.2
            # self.contact_filt = torch.logical_or(contact, self.last_contacts)
            self.contact_filt = torch.logical_or(contact, self.antidragging_last_contacts)
            # self.last_contacts = contact
            self.antidragging_last_contacts = contact
            # 更新足部触地时间
            # print("feet_contact_time: ", self.feet_contact_time[0])
            self.feet_contact_time += self.dt * contact
            self.feet_contact_time *= contact  # 如果当前帧没接触地面，则归零

            # 设定触地时间阈值
            contact_time_threshold = 1.3  # 设置最大触地时间（秒）
            # penalty_factor = 10.0  # 设定惩罚系数

            if torch.max(self.feet_contact_time) > contact_time_threshold:
            # 计算惩罚项（超过阈值的部分进行惩罚）
                penalty = torch.sum((self.feet_contact_time - contact_time_threshold).clamp(min=0), dim=1)
            else:
                penalty = 0
            # print("penalty: ", penalty)
            # 让静止机器人不受影响
            penalty *= torch.norm(self.commands[:, :2], dim=1) > 0.01  
            del contact
        return penalty  # 负奖励，越大越糟糕

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    # ------------newly added reward functions----------------
    def _reward_action_magnitude(self):
        return torch.sum(torch.square(torch.maximum(torch.abs(self.actions[:,[0,3,6,9]]) - 1.0,torch.zeros_like(self.actions[:,[0,3,6,9]]))), dim=1)
        # return torch.sum(torch.square(self.actions[:, [0, 3, 6, 9]]), dim=1)


    def _reward_power(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_power_distribution(self):
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_smoothness(self):
        return torch.sum(torch.square(self.last_last_actions - 2*self.last_actions + self.actions), dim=1)

    def _reward_clearance(self):
        # foot_pos = self.rigid_body_pos[:, self.feet_indices,:]
        #
        # foot_pos -= self.root_states[:,:3].unsqueeze(1)
        # foot_pos = torch.reshape(foot_pos,(self.num_envs * 4,-1))
        # foot_pos_base = quat_rotate_inverse(self.base_quat.repeat(4, 1), foot_pos)
        # foot_pos_base = torch.reshape(foot_pos_base,(self.num_envs,4,-1))
        # foot_heights = foot_pos_base[:,:,2]
        # if (self.common_step_counter <= 5) | (self.common_step_counter > self.max_episode_length):
        #     return torch.zeros(self.num_envs, device=self.device)

        if self.cfg.terrain.mesh_type == 'plane':
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
        else:
            points = self.rigid_body_pos[:, self.feet_indices,:]

            #measure ground height under the foot
            points += self.terrain.cfg.border_size
            points = (points / self.terrain.cfg.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights - self.cfg.asset.foot_radius

        foot_lateral_vel = torch.norm(self.rigid_body_lin_vel[:, self.feet_indices,:2], dim = -1)
        # print("max foot_heights: ", torch.max(foot_heights))
        # print("min foot_heights: ", torch.min(foot_heights))
        # print("foot_heights: ", torch.max(foot_heights[0]))
        # if torch.max(foot_heights[0]) >= 0.02:
        #     print("foot_heights: ", torch.max(foot_heights[0]), self.common_step_counter)
        # print("foot_lateral_vel: ", foot_lateral_vel[0])
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        # print("rew: ", torch.sum(foot_lateral_vel * (foot_heights - self.cfg.rewards.foot_height_target), dim = -1), self.common_step_counter)
        # return torch.sum(foot_lateral_vel * torch.abs(foot_heights - self.cfg.rewards.foot_height_target), dim = -1)
        return torch.sum(foot_lateral_vel * (foot_heights - self.cfg.rewards.foot_height_target), dim = -1)
        # return torch.sum(foot_lateral_vel * torch.square(foot_heights - self.cfg.rewards.foot_height_target), dim = -1)

    def _reward_foot_slippery(self):

        # foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
        foot_contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        # print("foot_contact_force: ", foot_contact_force[0][4])
        # 腿腾空时 foot_contact_force是否还有力 ？

        foot_lateral_vel = torch.norm(self.rigid_body_lin_vel[:, self.feet_indices,:2], dim = -1)
        # print("foot_lateral_vel: ", foot_lateral_vel[0][3])
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(foot_lateral_vel * torch.abs(foot_contact_force), dim = -1)
        
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0,3,6,9]] - self.default_dof_pos[:, [0,3,6,9]]), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        forward = quat_apply(self.base_quat, self.forward_vec)
        # print("forward: ", forward)
        # heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        # print("heading: ", heading)
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        # cheat_penalty[:self.roughflat_start_idx] = cheat
        cheat_penalty[:] = cheat
        # print("cheat_penalty: ", cheat_penalty)
        return cheat_penalty

    # def _reward_feet_edge(self):
    #     feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices,
    #                     :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
    #     feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
    #     feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
    #     feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

    #     self.feet_at_edge = self.contact_filt & feet_at_edge
    #     rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)

    #     edge_reward = torch.zeros_like(rew)
    #     edge_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
    #     return edge_reward

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        # rew = rew #* (self.terrain_levels > 3)

        # rew = rew.float()
        # stumble_reward = torch.zeros_like(rew)
        # # stumble_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        # stumble_reward = rew
        return rew #stumble_reward

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)

if __name__ == "__main__":
    cfg = MBRLHexapodCfg()
    sim_params = gymapi.SimParams()
    sim_params = {"sim": class_to_dict(cfg.sim)}
    args = get_args()
    sim_params = parse_sim_params(args, sim_params)
    physics_engine = gymapi.SIM_PHYSX
    sim_device = 'cuda'
    headless = False
    cfg.env.num_envs = 10
    cfg.depth.camera_num_envs = 10
    hexapod = HexapodRobot(cfg, sim_params, physics_engine, sim_device, headless)
    actions = torch.zeros(10, 18, device=hexapod.device, requires_grad=False)
    while True:
        hexapod.step(actions)