import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

class Robot:
    def __init__(self, dt):
        self.dt = dt
        # self.model_params = whatever, not used in this demo

    def f_kinematics(self, x, u):
        # TODO: assert x.type == u.type == casadi variable
        return ca.horzcat(x[0] + self.dt * x[1], x[1] + self.dt * u) # x[0] is position, x[1] is velocity, u is acceleration