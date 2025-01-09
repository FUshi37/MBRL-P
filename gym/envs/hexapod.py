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
from hexapod_robot_config import HexapodRobotCfg
from base_task import BaseTask
from gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params

def quat_to_euler(quat):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
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
        
        sim_params.use_gpu_pipeline = True
        if sim_device_type == 'cuda' :#and sim_params.use_gpu_pipeline:#
            self.device = self.sim_device
        else:
            self.device = 'cpu'#torch.device('cpu')#

        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = None
        
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        
        # # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)
        
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
        
        self._init_buffers()
        self.global_counter = 0
        
        if headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        
    
    def step(self, action):
        clip_actions = self.cfg.normalization.clip_actions
        if clip_actions:
            self.actions = torch.clip(action, -clip_actions, clip_actions).to(device=self.device)
        
        self.global_counter += 1
        
        # TODO: step simulation
        self.render()
        for _ in range(self.cfg.control.action_repeat):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # self.roll, self.pitch = quat_to_euler(self.base_quat)

        self.termination()
        self.calculate_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.get_observations()
        
        self.last_actions = self.actions
        
        # if self.viewer and self.enable_viewer_sync and self.debug_viz:
        #     self._draw_debug_viz()
        
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.infos
    
    # def _update_goals(self):
    #     next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
    #     self.cur_goal_idx[next_flag] += 1
    #     self.reach_goal_timer[next_flag] = 0

    #     self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
    #     self.reach_goal_timer[self.reached_goal_ids] += 1

    #     self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
    #     self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

    #     norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    #     target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    #     self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    #     norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
    #     target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
    #     self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
            
    def create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # terrain TODO
        self._create_ground_plane()
        self._create_envs()
    
    def _create_envs(self):
        
        
        asset_path = self.cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        # asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # asset_options.density = self.cfg.asset.density
        # asset_options.angular_damping = self.cfg.asset.angular_damping
        # asset_options.linear_damping = self.cfg.asset.linear_damping
        # asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # asset_options.armature = self.cfg.asset.armature
        # asset_options.thickness = self.cfg.asset.thickness
        # asset_options.disable_gravity = self.cfg.asset.disable_gravity
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        
        # # save body names from the asset
        # body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # self.num_bodies = len(body_names)
        # self.num_dofs = len(self.dof_names)
        # feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # penalized_contact_names = []
        # for name in self.cfg.asset.penalize_contacts_on:
        #     penalized_contact_names.extend([s for s in body_names if name in s])
        # termination_contact_names = []
        # for name in self.cfg.asset.terminate_after_contacts_on:
        #     termination_contact_names.extend([s for s in body_names if name in s])
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
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
        self.env_origins[:, 2] = 0.2      
        # self.env_origins = torch.zeros((self.num_envs, 3), device=self.device)
        # for i in range(self.num_envs):
        #     pos = self.gym.get_actor_root_state(self.sim, self.actor_handles[i])[0]
        #     self.env_origins[i] = torch.tensor(pos, device=self.device)
    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
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
    
    def calculate_reward(self):
        self.rew_buf = 0.
        
    
    def get_observations(self):
        # imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        # if self.global_counter % 5 == 0:
        #     self.delta_yaw = self.target_yaw - self.yaw
        #     self.delta_next_yaw = self.next_target_yaw - self.yaw
        # obs_buf = torch.cat((#skill_vector, 
        #                     self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
        #                     imu_obs,    #[1,2]
        #                     0*self.delta_yaw[:, None], 
        #                     self.delta_yaw[:, None],
        #                     self.delta_next_yaw[:, None],
        #                     0*self.commands[:, 0:2], 
        #                     self.commands[:, 0:1],  #[1,1]
        #                     (self.env_class != 17).float()[:, None], 
        #                     (self.env_class == 17).float()[:, None],
        #                     self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
        #                     self.reindex(self.dof_vel * self.obs_scales.dof_vel),
        #                     self.reindex(self.action_history_buf[:, -1]),
        #                     self.reindex_feet(self.contact_filt.float()-0.5),
        #                     ),dim=-1)
                            
        pass
    
    def termination(self):
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        pass
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            # for evt in self.gym.query_viewer_action_events(self.viewer):
            #     if evt.action == "QUIT" and evt.value > 0:
            #         sys.exit()
            #     elif evt.action == "toggle_viewer_sync" and evt.value > 0:
            #         self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device!= 'cpu':
                self.gym.fetch_results(self.sim, True)
            
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def _compute_torques(self, actions):
        # TODO
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        # return actions
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

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
        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()
        # self.measured_heights = 0
        
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # for i in range(self.num_dofs):
        #     name = self.dof_names[i]
        #     angle = self.cfg.init_state.default_joint_angles[name]
        #     self.default_dof_pos[i] = angle
        #     found = False
        #     for dof_name in self.cfg.control.stiffness.keys():
        #         if dof_name in name:
        #             self.p_gains[i] = self.cfg.control.stiffness[dof_name]
        #             self.d_gains[i] = self.cfg.control.damping[dof_name]
        #             found = True
        #     if not found:
        #         self.p_gains[i] = 0.
        #         self.d_gains[i] = 0.
        #         if self.cfg.control.control_type in ["P", "V"]:
        #             print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    
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
    actions = torch.zeros(1, 18, device=hexapod.device, requires_grad=False)
    while True:
        hexapod.step(actions)