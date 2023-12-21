import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from interface import Interface
from controllers.mpc_base import MPC_Base
from robot_models.base import Base
from robot_models.obstacles import Obstacles

dt = 0.1
N = 50
t_total = 50
x_start = np.array([0., 0., 0., 0., 0., 0.])     
x_target = np.array([50., 50., 0., 0., 0., 0.]) 
obstacle1 = Obstacles(25, 30, 6)
#obstacle2 = Obstacles(30, 30, 4)
base_demo = Base(dt)
mpc_demo = MPC_Base(base_demo, obstacle1, N=N)
# mpc_demo = MPC_Base(base_demo, N=N)
world = Interface(dt, t_total, x_start, x_target, mpc_demo, physical_sim=False)

world.run()
world.plot()