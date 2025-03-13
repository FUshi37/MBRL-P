import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import sys
from time import time
from warnings import WarningMessage

import copy
import math
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from gym import GYM_ENVS_DIR, GYM_ROOT_DIR
from .hexapod_robot_config import HexapodRobotCfg
# from gym.utils.terrain import Terrain
from .base_task import BaseTask
from gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
# MBRL
# import observation_buffer
from gym.envs import observation_buffer
# MBRL

import cv2
import json

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
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

class Hexapod():
    def __init__(self, cfg: HexapodRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        self.headless = headless
        self.debug_viz = True
        self._parse_cfg(self.cfg)
        
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        
        sim_params.use_gpu_pipeline = True
        if sim_device_type == 'cuda' :#and sim_params.use_gpu_pipeline:#
            self.device = self.sim_device
        else:
            self.device = 'cpu'#torch.device('cpu')#

        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1
        
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        print("num_obs: ", self.num_obs)
        print("num_priobs: ", self.num_privileged_obs)
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # MBRL
        self.include_history_steps = cfg.env.include_history_steps
        # self.height_dim = cfg.env.height_dim
        self.privileged_dim = cfg.env.privileged_dim
        
        # allocate buffers
        if cfg.env.include_history_steps is not None:
            self.obs_buf_history = observation_buffer.ObservationBuffer(
                self.num_envs, self.num_obs,
                self.include_history_steps, self.device)
        # MBRL
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
        
        self.infos = {}
        
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        
        self.enable_viewer_sync = True
        self.viewer = None
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        self._init_buffers()
        self._prepare_reward_function()
        self.global_counter = 0
        self.total_env_steps_counter = 0
        
        if headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            
        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
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
        
        self.ema_decay = 0.9
        self.velocity_ema = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.velocity_ema_list = []
        self.EMA_LEN = 50
        
        self.post_physics_step()
    
    def step(self, actions):
        # base task legged gym
        clip_actions = self.cfg.normalization.clip_actions
        if clip_actions:
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(device=self.device)
        # print("actions: ", self.actions)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        self.global_counter += 1
        
        # # 测试
        # if self.motorcommand_index == len(self.motorcommand) - 1:
        #     self.motorcommand_index = -1
        # if self.motorcommand_index < len(self.motorcommand) - 1:
        #     self.motorcommand_index = self.motorcommand_index + 1
        # self.actions = self.motorcommand[self.motorcommand_index]
        # # print("actions len: ", len(self.actions))
        # # self.actions[3:18] = [0] * 15 
        # # self.actions[1:18] = [0] * 17
        # # print("actions: ", self.actions)
        # self.actions = torch.tensor(self.actions, dtype=torch.float32).to(device=self.device)
        
        # self.actions.view(1, 18)
        # self.actions = self.actions.repeat(1, 1)
        
        # print("actions shape: ", self.actions.shape)
        
        # # extreme parkour
        # actions.to(self.device)
        # self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # if self.cfg.domain_rand.action_delay:
        #     if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
        #         if len(self.cfg.domain_rand.action_curr_step) != 0:
        #             self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
        #     if self.viewer:
        #         self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
        #     indices = -self.delay -1
        #     actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        # self.global_counter += 1
        # self.total_env_steps_counter += 1
        # clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # TODO: step simulation
        self.render()
        for _ in range(self.cfg.control.action_repeat):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # print("torques: ", self.torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        # self.post_physics_step
        # MBRL
        reset_env_ids, terminal_amp_states = self.post_physics_step()
        # MBRL
        
        
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # MBRL
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        # MBRL
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.infos["delta_yaw_ok"] = self.delta_yaw < 0.6
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.infos["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.infos["depth"] = None
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.infos
    
    def get_history_observations(self):
        return self.obs_history_buf
    
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)
    
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.roll, self.pitch, self.yaw = quat_to_euler(self.base_quat)
        self.yaw += math.pi/2
        # 使用torch.where处理所有环境的角度值
        self.yaw = torch.where(self.yaw > math.pi, self.yaw - 2 * math.pi, self.yaw)
        self.yaw = torch.where(self.yaw < -math.pi, self.yaw + 2 * math.pi, self.yaw)
        
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._update_goals()
        
        self.termination()
        current_vel_xy = self.root_states[:, 7:9].clone()
        # self.velocity_ema = self.ema_decay * self.velocity_ema + (1 - self.ema_decay) * current_vel_xy
        if len(self.velocity_ema_list) < self.EMA_LEN:
            self.velocity_ema_list.append(current_vel_xy)
        else:
            self.velocity_ema_list.pop(0)
            self.velocity_ema_list.append(current_vel_xy)
        self.velocity_ema = torch.stack(self.velocity_ema_list, dim=0).mean(dim=0)
        
        self.calculate_reward()
        # print("reward: ", self.rew_buf)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        # print("env_ids: ", env_ids)
        self.reset_idx(env_ids)
        
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        
        self.update_depth_buffer()
        
        self.compute_observations()
        
        self.last_actions = self.actions
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self._draw_height_samples()
            # self._draw_goals()
            # self._draw_feet()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
                
        return env_ids, terminal_amp_states
    
    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        # self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
            
    def create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id  # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # terrain TODO
        self._create_ground_plane()
        self._create_envs()
    
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov 
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)
            
            local_transform = gymapi.Transform()
            
            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0], config.angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    
    def _create_envs(self):
        
        
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
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        
        for s in ["l1_ft", "l2_ft", "l3_ft", "r1_ft", "r2_ft", "r3_ft"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
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
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            if isinstance(body_props, tuple):
                body_props = body_props[0]  # 从元组中取出第一个元素（列表）
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            self.attach_camera(i, env_handle, actor_handle)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    
    def set_camera(self, position, lookat):
        """Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.0     
        # self.env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        # for i in range(self.num_envs):
        #     pos = self.gym.get_actor_root_state(self.sim, self.actor_handles[i])[0]
        #     self.env_origins[i] = torch.tensor(pos, device=self.device)
        
        # self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # self.cur_goals = self._gather_cur_goals()
        # self.next_goals = self._gather_cur_goals(future=1)
        
        # max_init_level = self.cfg.terrain.max_init_terrain_level
        # if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
        # self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
        # self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        
        # self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
        # self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        # last_col = temp[:, -1].unsqueeze(1)
        # self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
        self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        reward_norm_factor = 1#np.sum(list(self.reward_scales.values()))
        for rew in self.reward_scales:
            self.reward_scales[rew] = self.reward_scales[rew] / reward_norm_factor
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # MBRL
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        # MBRL
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # curriculum terrain
        # TODO
        
        # reset robot states TODO
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)
        
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # reset buffers
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0
    
    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos #* torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)#
        self.dof_vel[env_ids] = 0.

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
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _prepare_reward_function(self):
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
    
    def calculate_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
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
        USE_MBRL = False
        if USE_MBRL == True:
            self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.commands[:, :3] * self.commands_scale,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions
                                        ),dim=-1)

            # if (self.cfg.env.privileged_obs):
                # # add perceptive inputs if not blind
                # if self.cfg.terrain.measure_heights:
                #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

                # if self.cfg.domain_rand.randomize_friction:
                #     self.privileged_obs_buf= torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)

                # if self.cfg.domain_rand.randomize_restitution:
                #     self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)

                # if (self.cfg.domain_rand.randomize_base_mass):
                #     self.privileged_obs_buf = torch.cat((self.randomized_added_masses ,self.privileged_obs_buf), dim=-1)

                # if (self.cfg.domain_rand.randomize_com_pos):
                #     self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos ,self.privileged_obs_buf), dim=-1)

                # if (self.cfg.domain_rand.randomize_gains):
                #     self.privileged_obs_buf = torch.cat(((self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains ,self.privileged_obs_buf), dim=-1)
                #     self.privileged_obs_buf = torch.cat(((self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains, self.privileged_obs_buf),
                #                                         dim=-1)

                # contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
                # self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1)
                # contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
                # self.privileged_obs_buf = torch.cat((contact_flag, self.privileged_obs_buf), dim=-1)

            # # add noise if needed
            # if self.add_noise:
            #     self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec


            # # Remove velocity observations from policy observation.
            # if self.num_obs == self.num_privileged_obs - 6:
            #     self.obs_buf = self.privileged_obs_buf[:, 6:]
            # elif self.num_obs == self.num_privileged_obs - 3:
            #     self.obs_buf = self.privileged_obs_buf[:, 3:]
            # else:
            #     self.obs_buf = torch.clone(self.privileged_obs_buf)
            self.obs_buf = torch.clone(self.privileged_obs_buf)
        else:
            imu_obs = torch.stack((self.roll, self.pitch), dim=1)
            if self.global_counter % 5 == 0:
                self.delta_yaw = self.target_yaw - self.yaw
                self.delta_next_yaw = self.next_target_yaw - self.yaw
            obs_buf = torch.cat((#skill_vector, 
                                self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                                imu_obs,    #[1,2]
                                0*self.delta_yaw[:, None], 
                                self.delta_yaw[:, None],
                                self.delta_next_yaw[:, None],
                                0*self.commands[:, 0:2], 
                                self.commands[:, 0:1],  #[1,1]
                                # (self.env_class != 17).float()[:, None], 
                                # (self.env_class == 17).float()[:, None],
                                # self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                                # self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                                # self.reindex(self.action_history_buf[:, -1]),
                                # self.reindex_feet(self.contact_filt.float()-0.5),
                                (self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos,
                                (self.dof_vel * self.obs_scales.dof_vel),
                                (self.action_history_buf[:, -1]),
                                (self.contact_filt.float()-0.5),
                                ),dim=-1)
            priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                    0 * self.base_lin_vel,
                                    0 * self.base_lin_vel), dim=-1)
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1
            ), dim=-1)
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
                self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            else:
                self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )
            # TODO: contact force
            # self.contact_buf = torch.where(
            #     (self.episode_length_buf <= 1)[:, None, None], 
            #     torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            #     torch.cat([
            #         self.contact_buf[:, 1:],
            #         self.contact_filt.float().unsqueeze(1)
            #     ], dim=1)
            # )              
            
            # pass
    
    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
    
    def get_observations(self):
        # MBRL
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        # MBRL
        return policy_obs
        # return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def termination(self):
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf
        
        # self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < -0.25

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff
        # print("reach_goal_cutoff: ", reach_goal_cutoff)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        # print("time_out_buf: ", self.time_out_buf)
        # print("roll_cutoff: ", roll_cutoff)
        # print("pitch_cutoff: ", pitch_cutoff)
        # print("height_cutoff: ", height_cutoff)
        pass
    
    def lookat(self, i):
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if not self.free_cam:
                self.lookat(self.lookat_id)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                
                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.2
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.2
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] += 0.5
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 3] -= 0.5
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()

                        
                
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
            
            if not self.free_cam:
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self.lookat_vec = cam_trans - look_at_pos
    

    def _compute_torques(self, actions):
        # TODO
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            # print("joint vel: ", self.dof_vel)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # 4.2
        
        # actions_scaled = actions * self.cfg.control.action_scale
        # control_type = self.cfg.control.control_type
        # if control_type=="P":
        #     if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
        #         torques = self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains*self.dof_vel
        #     else:
        #         torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        # elif control_type=="V":
        #     torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        # elif control_type=="T":
        #     torques = actions_scaled
        # else:
        #     raise NameError(f"Unknown controller type: {control_type}")
        # return torch.clip(torques, -self.torque_limits, self.torque_limits)
        # return actions
    
    # def _update_terrain_curriculum(self, env_ids):
    #     """ Implements the game-inspired curriculum.

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # # Implement Terrain curriculum
    #     # if not self.init_done:
    #     #     # don't change on initial reset
    #     #     return
        
    #     dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    #     threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
    #     move_up =dis_to_origin > 0.8*threshold
    #     move_down = dis_to_origin < 0.4*threshold

    #     self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    #     # # Robots that solve the last level are sent to a random one
    #     self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
    #                                                torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
    #                                                torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
    #     self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    #     self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    #     temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
    #     last_col = temp[:, -1].unsqueeze(1)
    #     self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
    #     self.cur_goals = self._gather_cur_goals()
    #     self.next_goals = self._gather_cur_goals(future=1)

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_cnt = self.gym.get_sim_force_sensor_count(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self._reset_root_states(torch.arange(self.num_envs, device=self.device))
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 6, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # # initialize some data used later on
        self.common_step_counter = 0
        self.infos = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # 固定目标速度
        self.commands[:, 0] = 0.0  # x方向速度
        self.commands[:, 1] = 1.0  # y方向速度
        self.commands[:, 2] = 0.0  # yaw角速度
        
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 4, device=self.device, dtype=torch.float)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            # for dof_name in self.cfg.control.stiffness.keys():
            #     if dof_name in name:
            #         self.p_gains[i] = self.cfg.control.stiffness[dof_name]
            #         self.d_gains[i] = self.cfg.control.damping[dof_name]
            #         found = True
            # if not found:
            #     self.p_gains[i] = 0.
            #     self.d_gains[i] = 0.
            #     if self.cfg.control.control_type in ["P", "V"]:
            #         print(f"PD gain of joint {name} were not defined, setting them to zero")
            self.p_gains[i] = self.cfg.control.stiffness['joint']
            self.d_gains[i] = self.cfg.control.damping['joint']
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        
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
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
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
                # props[i]["driveMode"] = gymapi.DOF_MODE_EFFORT  # 设置为力控模式
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # print("torque limits: ", self.torque_limits)
        return props

    def _process_rigid_body_props(self, props, env_id):
        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params
    
    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)
    
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
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points
    
    # def _draw_goals(self):
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
    #     sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
    #     sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
    #     goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
    #     for i, goal in enumerate(goals):
    #         goal_xy = goal[:2] + self.terrain.cfg.border_size
    #         pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
    #         goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
    #         pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
    #         if i == self.cur_goal_idx[self.lookat_id].cpu().item():
    #             gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #             if self.reached_goal_ids[self.lookat_id]:
    #                 gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    #         else:
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    #     if not self.cfg.depth.use_camera:
    #         sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
    #         pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
    #         for i in range(5):
    #             norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #             target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #             pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
    #             pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
    #             gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
    #         sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
    #         for i in range(5):
    #             norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
    #             target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
    #             pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
    #             pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
    #             gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
    # def _draw_feet(self):
    #     if hasattr(self, 'feet_at_edge'):
    #         non_edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0))
    #         edge_geom = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0, 0))

    #         feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
    #         for i in range(4):
    #             pose = gymapi.Transform(gymapi.Vec3(feet_pos[self.lookat_id, i, 0], feet_pos[self.lookat_id, i, 1], feet_pos[self.lookat_id, i, 2]), r=None)
    #             if self.feet_at_edge[self.lookat_id, i]:
    #                 gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[i], pose)
    #             else:
    #                 gymutil.draw_lines(non_edge_geom, self.gym, self.viewer, self.envs[i], pose)
    
    # def _reward_tracking_goal_vel(self):
    #     norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #     target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #     # print("self.commands: ", self.commands)
    #     cur_vel = self.root_states[:, 7:9]
    #     # print("cur_vel: ", cur_vel)
    #     rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
    #     # print("rew: ", rew)
    #     return rew

    # def _reward_tracking_goal_vel(self):
    #     # 从 commands 中提取目标速度的 x-y 分量（假设 commands 的前两列是目标线速度）
    #     target_vel_xy = self.commands[:, 0:2]  # shape: (num_envs, 2)
        
    #     # 计算目标速度的模长（目标速度大小）
    #     target_speed = torch.norm(target_vel_xy, dim=-1, keepdim=True)  # shape: (num_envs, 1)
        
    #     # 归一化目标方向（如果目标速度为 0，则方向为 0）
    #     target_dir = target_vel_xy / (target_speed + 1e-5)  # shape: (num_envs, 2)
        
    #     # 获取当前基座的 x-y 线速度
    #     current_vel_xy = self.root_states[:, 7:9]  # shape: (num_envs, 2)
    #     # print("current_vel_xy: ", current_vel_xy)
        
    #     # # 计算当前速度在目标方向上的投影（点积）
    #     # vel_projection = torch.sum(target_dir * current_vel_xy, dim=-1)  # shape: (num_envs,)
        
    #     # 使用 EMA 速度计算投影
    #     vel_projection = torch.sum(target_dir * self.velocity_ema, dim=-1)
    #     print("ema vel: ", self.velocity_ema)
        
    #     # 归一化奖励：投影值 / 目标速度大小（当目标速度为 0 时奖励为 0）
    #     rew = torch.where(
    #         target_speed.squeeze() > 1e-3,  # 判断目标速度是否非零
    #         vel_projection / (target_speed.squeeze() + 1e-5),
    #         torch.zeros_like(vel_projection)
    #     )
    #     # print("rew: ", rew)
    #     return rew
    
    def _reward_tracking_goal_vel(self):
        #     # 从 commands 中提取目标速度的 x-y 分量（假设 commands 的前两列是目标线速度）
        target_vel_x = self.commands[:, 0:1]  # shape: (num_envs, 2)
        target_vel_y = self.commands[:, 1:2]
        # current_vel_x = self.root_states[:, 7:8]  # shape: (num_envs, 2)
        # current_vel_y = self.root_states[:, 8:9]
        current_vel_x = self.velocity_ema[:, 0:1]
        current_vel_y = self.velocity_ema[:, 1:2]
        # print("target_vel_x: ", target_vel_x)
        # print("current_vel_x: ", current_vel_x)
        # print("target_vel_y: ", target_vel_y)
        # print("current_vel_y: ", current_vel_y)
        
        rew_x = torch.exp(-torch.abs(target_vel_x - current_vel_x)/0.05)
        rew_y = torch.exp(-torch.abs(target_vel_y - current_vel_y)/0.05)
        rew = (rew_x + rew_y).reshape(-1)
        # print("rew velocity tracking: ", rew)
        # rew = 0
        return rew
    
    def _reward_tracking_yaw(self):
        self.target_yaw = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # print("yaw: ", self.yaw)
        # print("target_yaw: ", self.target_yaw)
        # print("rew: ", rew)
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        # rew[self.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # rew[self.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    # def _reward_hip_pos(self):
    #     return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    # def _reward_feet_edge(self):
    #     feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
    #     feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
    #     feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
    #     feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
    #     self.feet_at_edge = self.contact_filt & feet_at_edge
    #     rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    #     return rew
    
if __name__ == "__main__":
    cfg = HexapodRobotCfg()
    sim_params = gymapi.SimParams()
    sim_params = {"sim": class_to_dict(cfg.sim)}
    args = get_args()
    sim_params = parse_sim_params(args, sim_params)
    physics_engine = gymapi.SIM_PHYSX
    sim_device = 'cuda'
    headless = False
    hexapod = Hexapod(cfg, sim_params, physics_engine, sim_device, headless)
    actions = torch.zeros(100, 18, device=hexapod.device, requires_grad=False)
    while True:
        hexapod.step(actions)