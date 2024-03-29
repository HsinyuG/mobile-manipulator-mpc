import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class Base:
    def __init__(self, dt):
        self.dt = dt
        self.base_length = 2 * (0.7/2 + 0.157) # estimated in simulation  
        self.base_width = 0.52 # estimated in simulation
          
    def base_radius(self):
        # base_radius = math.sqrt((self.base_length / 2.)**2 + (self.base_width / 2.) ** 2)
        # return base_radius
        return 0.4 # 0.57
    
    def f_kinematics(self, x, u, limited_yaw=False):
        # TODO: assert x.type == u.type == casadi variable
        x_next = ca.horzcat(
            x[0] + self.dt * x[3],
            x[1] + self.dt * x[4],
            x[2] + self.dt * x[5], 
            x[3] + self.dt * (u[0]*np.cos(x[2])- x[4]*x[5]), #- x[4] * x[5] #- x[3] * np.tan(x[2]) * x[5]),
            x[4] + self.dt * (u[0]*np.sin(x[2]) + x[3]*x[5]), #+ x[3] * x[5]),
            x[5] + self.dt * u[1]
        ) # x[0] = x, x[1] = y, x[2] = psi, x[3] = x_dot, x[4] = y_dot, x[5] = psi_dot, u[0] = v_dot, u[1] = w_dot

        if limited_yaw:
            x_next[2] = ca.fmod((x_next[2] + ca.pi), (2*ca.pi)) - ca.pi # to [-pi, pi)
        
        return x_next