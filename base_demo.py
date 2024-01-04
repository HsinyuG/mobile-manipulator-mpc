import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
# from interface_base_nosim import Interface
from interface_base import Interface
from controllers.mpc_base import MPC_Base
from robot_models.base import Base
from robot_models.obstacles import Obstacles

dt = 0.1
N = 50 # TODO: 100 no solution, maybe due to error in terminatinf traj ref
t_total = 5 # desired completion time, consider v_max = 2m/s
x_start = np.array([0., 0., 0., 0., 0., 0.])     
x_target = np.array([5., 5., 0., 0., 0., 0.]) 

obstacle_list = [
    Obstacles(2.5, 3.0, 0.6),
    Obstacles(2.5, 1.0, 0.6)
]
#obstacle2 = Obstacles(30, 30, 4)
base_demo = Base(dt)
mpc_demo = MPC_Base(base_demo, obstacle_list, N=N)
# mpc_demo = MPC_Base(base_demo, N=N)
world = Interface(dt, t_total, x_start, x_target, mpc_demo, physical_sim=True)

world.run()
world.plot2D()