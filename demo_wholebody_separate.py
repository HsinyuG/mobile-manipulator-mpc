import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from interface_wholebody_separate import Interface

from controllers.mpc_manipulator_3DoF import MPCManipulator3DoF
from robot_models.manipulator_3DoF import ManipulatorPanda3DoF

from controllers.mpc_base import MPCBase
from robot_models.base import Base
from robot_models.obstacles import Obstacles

# move base
dt = 0.1
N = 50
t_move = 5 # desired completion time, consider v_max = 2m/s
t_manipulate = 2
base_x_start = np.array([0., 0., ca.pi/4, 0., 0., 0.])  # ca.pi/4
joint_x_start = np.array([0, -3, 0]) # 0 0 0
global_pose_target = np.array([5.4243, 5.4243, 0.606+0.333+0.5, ca.pi/4])

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    Obstacles(2.5, 1.0, 0.6),
    Obstacles(5.4243, 5.4243, 0.1)
]

robot_base = Base(dt)
controller_base = MPCBase(robot_base, obstacle_list, N=N)
robot_manipulator = ManipulatorPanda3DoF(dt)
controller_manipulator = MPCManipulator3DoF(robot_manipulator, N=N)

world = Interface(dt, t_move, t_manipulate, base_x_start, joint_x_start, global_pose_target, controller_base, controller_manipulator, physical_sim=True)

world.run()
world.plot3D()
