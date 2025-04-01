# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import glob

from gym.envs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class MBRLHexapodCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 18 + 18 + 3 + 3 + 3 # 45 # proprioception / dof_pod dof_vel base_ag_vel projected_gravity commands
        action_dim = 18
        privileged_dim = 3 + 1 + 1 + 1 + 3 + 36 + 18 #+ 12 # 63 + 12 # base_lin_vel rand_friction rand_restitution rand_base_mass rand_com_pos rand_gains contact_force  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        forward_height_dim = 525 # for depth image prediction
        if not LeggedRobotCfg.terrain.is_plane:
            height_dim = 108 #273  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        if LeggedRobotCfg.terrain.is_plane:
            height_dim = 0  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim # 45 + 63 + 187 + 18 = 313
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        episode_length_s = 20
        # amp_motion_files = MOTION_FILES
        
        # n_scan = 132
        # n_priv = 3+3 +3
        # n_priv_latent = 4 + 1 + 12 +12
        # # n_proprio = 3 + 2 + 3 + 4 + 36 + 5
        # n_proprio = 65

    class terrain:
        is_plane = False
        mesh_type = 'trimesh'#'trimesh'#'plane'  # "heightfield" # none, plane, heightfield or trimesh or staircase
        horizontal_scale = 0.03  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True # True
        curriculum_counter = 10
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True #True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        #                      0.8]  # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_x = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2] # 1mx1.6m rectangle (without center line) # 9 13 
        measured_points_y = [-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2] # 12 21 
        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                     2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


        selected = False#False  # select a unique terrain type and pass all arguments
        # terrain_kwargs = None  # Dict of arguments for selected terrain
        terrain_kwargs = {'type': 'random_box_terrain', 'grid_size': 0.3, 'min_height': -0.5, 'max_height': 0.5}
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 4  # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        # terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.00#0.75  # slopes above this threshold will be corrected to vertical surfaces

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.1]  # x,y,z [m]
        default_joint_angles = { # target angles when action = 0.0
            "l1_bc": 0.,
            "l2_bc": 0.,
            "l3_bc": 0.,
            "r1_bc": 0.,
            "r2_bc": 0.,
            "r3_bc": 0.,
            
            "l1_cf": 0.,
            "l2_cf": 0.,
            "l3_cf": 0.,
            "r1_cf": 0.,
            "r2_cf": 0.,
            "r3_cf": 0.,
            
            "l1_ft": 0.,
            "l2_ft": 0.,
            "l3_ft": 0.,
            "r1_ft": 0.,
            "r2_ft": 0.,
            "r3_ft": 0.,}

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 5}  # [N*m/rad] #2.
        damping = {'joint': 0.0147}  # [N*m*s/rad] # 0.0047
        # stiffness = {'joint': 10.}  # [N*m/rad]
        # damping = {'joint': 0.0247}  # [N*m*s/rad] # 0.0247
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.15#0.15
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4


    class depth:
        use_camera = False
        camera_num_envs = 4096# 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        y_angle = [-5, 5]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/gym/assets/urdf/neuroant.urdf"
        foot_name = "foot_tip"
        hip_name = "bc"
        penalize_contacts_on = [] # ["thigh", "calf"]
        # penalize_contacts_on = ["l1_bc", "l1_cf", "l2_bc", "l2_cf", "l3_bc", "l3_cf", "r1_bc", "r1_cf", "r2_bc", "r2_cf", "r3_bc", "r3_cf"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        foot_radius = 0.0079

    class domain_rand:
        randomize_friction = True
        friction_range = [0.9, 1.5] # 2.0
        randomize_restitution = True
        restitution_range = [0.0, 0.0]

        randomize_base_mass = True
        added_mass_range = [0., 0.3]  # kg
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.02, 0.02] # 0.05
        com_y_pos_range = [-0.02, 0.02] # 0.05
        com_z_pos_range = [-0.00, 0.00] # 0.05

        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 0.02

        randomize_gains = True
        stiffness_multiplier_range = [0.95, 1.05] # 0.8 1.2
        damping_multiplier_range = [1.0, 1.0] # 0.8 1.2
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.00, 0.005]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            # privileged
            height_measurements = 5.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5


        clip_observations = 100.
        clip_actions = 6.0

        base_height = 0.08 # base height of A1, used to normalize measured height


    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0  # set lin_vel as privileged information
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0  # only for critic

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = False # True
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        soft_dof_pos_limit = 0.9
        base_height_target = 0.08
        foot_height_target = 0.02#0.03
        tracking_sigma = 0.01  # tracking reward = exp(-error^2/sigma)
        lin_vel_clip = 0.02#0.1

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 0.0#1.5
            tracking_lin_vel_x = 2.0
            tracking_lin_vel_y = 8.0
            tracking_ang_vel = 5.0#2.0
            ang_vel_xy = -0.75#-0.25
            torques = -0.0001#0.0001
            dof_acc = -2.5e-7
            base_height = -5.0 #-5.0#-7.5
            feet_air_time = 1.0 # 4.0
            no_feet_air_time = -1.0
            penalize_negative_force = -0.1
            # feet_air_time_tripod = 0.
            anti_dragging = -0.5#-0.5
            collision = -1.0
            feet_stumble = -7.5#5.0#-1.0
            action_rate = -0.01#-0.03
            clearance = 7.5#5.0#-3.0
            smoothness = -0.01#-0.01
            # feet_edge = -1.0
            dof_error = 0#-0.04
            negative_vel_y = -20.0
            lin_vel_z = -1.0
            cheat = -1.#-1
            stuck = -1
            
            foot_slippery = -0.3#-0.1
            
            dof_pos_limits = -1.0
            dof_vel_limits = -5.0
            torque_limits = -5.0
            
            hip_phase = 0
            
            x_offset_penalty = -0.0

    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0

        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.0, 0.0]  # min max [m/s]
            lin_vel_y = [0.055, 0.055]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0., 0.]

            flat_lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            flat_lin_vel_y = [0.055, 0.055]  # min max [m/s]
            flat_ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class MBRLHexapodCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32 # 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        scan_encoder_dims = [128, 64, 32]
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.00005
        # vel_predict_coef = 1.0
        # amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'flat_push1'
        experiment_name = 'hexapod_example'
        # algorithm_class_name = 'AMPPPO'
        algorithm_class_name = 'PPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000  # number of policy updates
        save_interval = 1000

        amp_reward_coef = 0.5 * 0.02  # set to 0 means not use amp reward
        # amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 6

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100

    class depth_encoder:
        if_depth = MBRLHexapodCfg.depth.use_camera
        depth_shape = MBRLHexapodCfg.depth.resized
        buffer_len = MBRLHexapodCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = MBRLHexapodCfg.depth.update_interval * 24

    # class estimator:
    #     train_with_estimated_states = True
    #     learning_rate = 1.e-4
    #     hidden_dims = [128, 64]
    #     priv_states_dim = MBRLHexapodCfg.env.n_priv
    #     num_prop = MBRLHexapodCfg.env.n_proprio
    #     num_scan = MBRLHexapodCfg.env.n_scan
