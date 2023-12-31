from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle

from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.dynamic_box_obstacle import DynamicBoxObstacle

from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle
from mpscenes.obstacles.dynamic_cylinder_obstacle import DynamicCylinderObstacle

from mpscenes.obstacles.urdf_obstacle import UrdfObstacle
from mpscenes.obstacles.dynamic_urdf_obstacle import DynamicUrdfObstacle

import gymnasium as gym
import numpy as np

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv


def create_obstacle(type, geometry, dynamic=False, movable=False, id="", rgba=[0.0, 0.0, 0.0, 1.0]):
    obstacle_classes = {
        ("sphere", False): SphereObstacle,
        ("sphere", True): DynamicSphereObstacle,
        ("box", False): BoxObstacle,
        ("box", True): DynamicBoxObstacle,
        ("cylinder", False): CylinderObstacle,
        ("cylinder", True): DynamicCylinderObstacle,
        ("urdf", False): UrdfObstacle,
        ("urdf", True): DynamicUrdfObstacle,
    }
    
    obstacle_class = obstacle_classes.get((type, dynamic), None)

    obstacle_dict = {
        "type": type,
        "geometry": geometry,
        "movable": movable,
        "rgba": rgba
    }

    if obstacle_class:
        obstacle = obstacle_class(name=id, content_dict=obstacle_dict)
    else:
        raise ValueError("Unsupported obstacle type or dynamic setting")

    return obstacle

# Only included to test this code
# Should not be included in the final version
def run_albert(obstacles, n_steps=1000, render=False, reconfigure_camera=False, goal=True, mode='vel'):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode=mode,
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0, # Change initial direction
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    
    action = np.zeros(env.n())
    action[0] = 0.2 #forward velocity
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    for obstacle in  obstacles:
        env.add_obstacle(obstacle)

    if reconfigure_camera:
        env.reconfigure_camera(4.0, 0.0, -90.01, (0, -2.0, 0))
        
    for _ in range(n_steps):
        ob, *_ = env.step(action)
    env.close()

# Only included to test this code
# Should not be included in the final version
if __name__ == "__main__":
    descriptions = [
        {
            "type": "sphere",
            "geometry": {"position": [1.0, -2.0, 2.0], "radius": 0.2},
            "rgba": [0.3, 0.5, 0.6, 1.0],
            'movable': True,},
        {
            "type": "cylinder",
            "movable": False,
            "geometry": {
                "position": [-1.0, -3.0, 0.0],
                "radius": 0.3,
                "height": 2.0,
            },
            "rgba": [0.1, 0.3, 0.3, 1.0],
        },
        {
            'type': 'box',
            'geometry': {
                'position' : [0.5, -2.0, 0.5],
                'orientation': [0.923, 0, 0, -0.38],
                'width': 0.5,
                'height': 0.5,
                'length': 0.5,
            },
            'movable': False,
        },
        {
            'type': 'box',
            'geometry': {
                'position' : [0.5, -0.5, 1.5],
                'orientation': [0.923, 0, 0, -0.38],
                'width': 0.5,
                'height': 0.5,
                'length': 0.5,
            },
            'movable': True,
        },
        {
            "type": "sphere",
            "geometry": {"trajectory": ["2.0 - 0.5 * t", "-0.0", "0.1"], "radius": 0.2},
            "movable": False,
            "dynamic": True,
            "rgba": [0.6, 0.6, 0.6, 1.0],
        }
    ]

    obstacles = [create_obstacle(**description) for description in descriptions]

    run_albert(obstacles, render=True)
