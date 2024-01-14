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
experiment_scenario = 1 # 1 or 2 or 0, 0 for debug

""" see `reconfigure_camera` in simulation \albert_robot.py to change view in simulation """

if experiment_scenario == 1:
    x_start = np.array([0, 0, 0, 0, 0, 0, -ca.pi/4, -ca.pi, ca.pi])     # x y psi dx dy dpsi q1 q2 q3
    global_pose_target = np.array([5-0.6, 5, 0.606+0.333+0.5, -ca.pi]) # want base stop at 5,5
    obstacle_manipulation_list = [
        (np.array([5.007-0.43, 5, 0.27+0.606+0.333]), np.array([[0, 0, -1]])), # relative 0.33, 0, 0.27, normal downwards
        (np.array([5.007-0.43, 5, 0.27+0.606+0.333]), np.array([[-1, 0, 0]])), # relative also 0.33, 0, 0.27, normal forwards
        (np.array([5.007-0.43, 5, 0.27+0.606+0.333]), np.array([[0, 1, 0]])), # relative also 0.33, 0, 0.27, normal rightwards
    ]

elif experiment_scenario == 2:
    x_start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    global_pose_target = np.array([5-0.6, 5, 0.606+0.333+0.5, -ca.pi]) # want base stop at 5,5
    obstacle_manipulation_list = [
        (np.array([2.5, 2, 0.35+0.606+0.333]), np.array([[1/ca.sqrt(2), 0, 1/ca.sqrt(2)]])),
        (np.array([2.5, 2, 0.35+0.606+0.333]), np.array([[-1/ca.sqrt(2), 0, 1/ca.sqrt(2)]])),
    ]

else: # debug
    x_start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    global_pose_target = np.array([-0.6, 0, 0.606+0.333+0.5, -ca.pi])
    obstacle_manipulation_list = []

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    Obstacles(2.5, 1.0, 0.6),
    Obstacles(5-0.6, 5, 0.1)
]

mobile_manipulator = MobileManipulator(dt)
mpc_controller = MPCWholeBody(mobile_manipulator, obstacle_list, obstacle_manipulation_list, N=N)
world = Interface(dt, t_move, t_manipulate, x_start, global_pose_target, mpc_controller, physical_sim=True)

world.run()
world.plot3D()