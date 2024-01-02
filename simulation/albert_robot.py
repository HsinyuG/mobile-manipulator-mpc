import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from simulation.obstacles import (
    movable_sphere1,
    static_box1,
    movable_box1,
    static_cylinder
)

def setup_environment(
        render=False, 
        reconfigure_camera=False, 
        goal=True, 
        obstacles=True, 
        mode='vel',
        initial_state=None,
        dt=0.01
    ):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode=mode,
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0, # Change initial direction
            facing_direction = 'x',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=dt, robots=robots, render=render
    )
    
    ob = env.reset(
        pos = initial_state # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]
    )
    print(f"Initial observation : {ob}")

    if obstacles:
        env.add_obstacle(movable_sphere1)
        env.add_obstacle(static_box1)
        env.add_obstacle(movable_box1)
        env.add_obstacle(static_cylinder)
    
    if reconfigure_camera:
        env.reconfigure_camera(4.0, 0.0, -90.01, (0, -2.0, 0))
    
    return env, ob


def run_step(env, action):
    ob, *_ = env.step(action)

    return ob
