import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from interface_wholebody_qref import Interface
from controllers.mpc_wholebody_qref import MPCWholeBody
from robot_models.mobile_manipulator import MobileManipulator
from robot_models.obstacles import Obstacles

dt = 0.1
N = 20 
t_move = 5 
t_manipulate = 2
x_start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])     # x y psi dx dy dpsi q1 q2 q3
# global_pose_target = np.array([5+0.6/ca.sqrt(2), 5+0.6/ca.sqrt(2), 0.606+0.333+0.5, ca.pi/4]) 
# global_pose_target = np.array([4+0.6/ca.sqrt(2), 4+0.6/ca.sqrt(2), 0.606+0.333+0.5, -ca.pi]) 
# global_pose_target = np.array([-0.6, 0, 0.606+0.333+0.5, -ca.pi]) 
global_pose_target = np.array([5-0.6, 5, 0.606+0.333+0.5, -ca.pi]) 

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    Obstacles(2.5, 1.0, 0.6),
    Obstacles(5-0.6, 5, 0.1)
]

mobile_manipulator = MobileManipulator(dt)
mpc_controller = MPCWholeBody(mobile_manipulator, obstacle_list, N=N)
world = Interface(dt, t_move, t_manipulate, x_start, global_pose_target, mpc_controller, physical_sim=True)

world.run()
world.plot3D()