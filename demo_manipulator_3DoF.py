import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from interface_manipulator import Interface
from controllers.mpc_manipulator_3DoF import MPCManipulator3DoF
from robot_models.manipulator_3DoF import ManipulatorPanda3DoF

dt = 0.1
N = 10
t_total = 2
x_start = np.array([0., 0., 0.])      # q1 q2 q3
pose_target = np.array([0.5, 0., 0.5])   # relative position to base frame of manipulator

bot_demo = ManipulatorPanda3DoF(dt)
mpc_demo = MPCManipulator3DoF(bot_demo, N=N)
world = Interface(dt, t_total, x_start, pose_target, mpc_demo, physical_sim=True)

world.run()
world.plotManipulator()