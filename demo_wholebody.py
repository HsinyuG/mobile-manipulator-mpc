import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from interface_wholebody import Interface
from controllers.mpc_wholebody import MPCWholeBody
from robot_models.mobile_manipulator import MobileManipulator
from robot_models.obstacles import Obstacles

dt = 0.1
N = 50 
t_total = 5 
x_start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])     
pose_target = np.array([5, 5, 1.106, ca.pi/4]) 

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    # Obstacles(2.5, 1.0, 0.6),
    Obstacles(5, 5, 0.1)
]

mobile_manipulator = MobileManipulator(dt)
mpc_controller = MPCWholeBody(mobile_manipulator, obstacle_list, N=N)
world = Interface(dt, t_total, x_start, pose_target, mpc_controller, physical_sim=True)

world.run()
world.plot3D()