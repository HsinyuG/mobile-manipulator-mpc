import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
# from interface_base_nosim import Interface
from interface_base import Interface
from controllers.mpc_base import MPCBase
from robot_models.base import Base
from robot_models.obstacles import Obstacles

dt = 0.1
N = 50
t_total = 5 # desired completion time, consider v_max = 2m/s
x_start = np.array([0., 0., ca.pi/4, 0., 0., 0.])  # ca.pi/4   
x_target = np.array([5., 5., -ca.pi/2, 0., 0., 0.]) # -ca.pi or -ca.pi/2

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    Obstacles(2.5, 1.0, 0.6)
]
#obstacle2 = Obstacles(30, 30, 4)
base_demo = Base(dt)
mpc_demo = MPCBase(base_demo, obstacle_list, N=N)
# mpc_demo = MPCBase(base_demo, N=N)
world = Interface(dt, t_total, x_start, x_target, mpc_demo, physical_sim=True)

world.run()
world.plot2D()