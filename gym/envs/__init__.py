
# from ...isaacgym.python import isaacgym
import isaacgym
from gym.envs import *
from gym import *
from gym.envs.hexapodTS import Hexapod
from gym.envs.hexapod_robot_config import HexapodRobotCfg, HexapodRobotCfgPPO
from gym.envs.hexapodMBRL import HexapodRobot
from gym.envs.hexapodMBRL_config import MBRLHexapodCfg, MBRLHexapodCfgPPO

import os

from gym.utils.task_registry import task_registry

GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GYM_ENVS_DIR = os.path.join(GYM_ROOT_DIR, "envs")

# task_registry.register( "hexapod", Hexapod, HexapodRobotCfg, HexapodRobotCfgPPO)
# MBRL
task_registry.register( "hexapod", HexapodRobot, MBRLHexapodCfg, MBRLHexapodCfgPPO)