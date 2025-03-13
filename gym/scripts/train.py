import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import sys

import numpy as np
from datetime import datetime

import isaacgym
import rsl_rl
from gym.envs import *
from gym.utils import get_args, task_registry
from shutil import copyfile
import torch
import wandb

def train(args):
    args.headless = True
    log_pth = GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
    wandb.init(project=args.proj_name, name=args.exptid, entity="fushi37", group=args.exptid[:3], mode=mode, dir="../../logs")
    wandb.save(GYM_ENVS_DIR + "/hexapod_robot_config.py", policy="now")
    wandb.save(GYM_ENVS_DIR + "/hexapodTS.py", policy="now")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    # ppo_runner, train_cfg = task_registry.make_wmp_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()
    train(args)
