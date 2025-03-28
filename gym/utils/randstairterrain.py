import numpy as np
import random
from gym.utils.trimesh import box_trimesh, combine_trimeshes, move_trimesh

class BoxCfg:
    mean_height = 0.6
    mean_width = 0.5
    
    height_ranges = [0.1, 1.1]
    width_ranges = [0.5, 0.7]
    
    pass

def generate_box(box_length, box_width, box_height, center_position):
    return box_trimesh(
        size=[box_length, box_width, box_height],
        center_position=center_position
    )

def generate_boxes_terrain(env_length, env_width, cfg: BoxCfg):
    num_rows = int(env_length // cfg.mean_width)
    num_cols = int(env_width // cfg.mean_width)
    
    all_boxes = []
    for i in range(num_rows):
        for j in range(num_cols):
            box_length = cfg.mean_width
            box_width = cfg.mean_width
            box_height = random.uniform(cfg.height_ranges[0], cfg.height_ranges[1])
            center_position = [i * box_length, j * box_width, box_height / 2]
            
            box = generate_box(box_length, box_width, box_height, center_position)
            all_boxes.append(box)
    
    terrain = all_boxes[0]
    for box in all_boxes[1:]:
        terrain = combine_trimeshes(terrain, box)
    
    return terrain