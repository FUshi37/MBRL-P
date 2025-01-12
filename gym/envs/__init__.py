
# from ...isaacgym.python import isaacgym
import isaacgym
from gym.envs import *
from gym import *

import os

GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GYM_ENVS_DIR = os.path.join(GYM_ROOT_DIR, "envs")