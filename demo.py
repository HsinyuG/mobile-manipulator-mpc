import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from interface import Interface
from controllers.mpc_demo import MPC
from robot_models.robot_demo import Robot

dt = 0.1
N = 10
t_total = 50
x_start = np.array([0., 0.])      # 1D position and velocity
x_target = np.array([50., 0.]) 

bot_demo = Robot(dt)
mpc_demo = MPC(bot_demo, N=N)
world = Interface(dt, t_total, x_start, x_target, mpc_demo, physical_sim=False)

world.run()
world.plot1D()