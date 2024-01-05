import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# only works when this script is imported outside robot_models package
from robot_models.manipulator_3DoF import ManipulatorPanda3DoF 
from robot_models.base import Base

class MobileManipulator:
    def __init__(self, dt):
        self.dt = dt
        self.base = Base(dt)
        self.manipulator = ManipulatorPanda3DoF(dt)
        self.baselink2joint1_x = 0 # 0.007
        self.baselink2joint1_z = 0.606 + 0.333

    def forward_tranformation(self, state):
        """
        description: 
        - calculate endpoint pose and joint positions by current state
        input: 
        - state: base [x, y, psi, dx, dy, dpsi], and manipulator [q1, q2, q3]
        output: 
        - pose_endpoint: manipulator [x, y, z, psi] in global coordinate
        - pos_joint_2: joint 2 [x, y, z] in global coordinate
        - pos_joint_3: joint 3 [x, y, z] in global coordinate
        """
        x = state[:6]
        q = state[6:]
        pos_endpoint, pos_joint_2, pos_joint_3 = self.manipulator.forward_tranformation(q)
        try:
            assert pos_endpoint[1] == 0.0
            # print('assert happens======================') it will happen when the variable has value
        except: pass

        pose_endpoint = ca.horzcat(
            x[0] + (pos_endpoint[0] + self.baselink2joint1_x) * ca.cos(x[2]),
            x[1] + (pos_endpoint[0] + self.baselink2joint1_x) * ca.sin(x[2]),
            0 + pos_endpoint[2] + self.baselink2joint1_z,
            x[2]
        )

        return pose_endpoint, pos_joint_2, pos_joint_3

    def f_kinematics(self, x, u):
        """
        description: 
        - calculate next state by current state and input
        input: 
        - x: base [x, y, psi, dx, dy, dpsi], and manipulator [q1, q2, q3]
        - u: base [dV, dw] and manipulator [dq1, dq2, dq3]
        output: 
        - x_q_next: [x, y, psi, dx, dy, dpsi, q1, q2, q3] of base and manipulator
        """
        x_base = x[:6]
        q = x[6:]
        u_base = u[:2]
        q_dot = u[2:]

        q_next = self.manipulator.f_kinematics(q, q_dot)
        x_base_next = self.base.f_kinematics(x_base, u_base)
        x_q_next = ca.horzcat(x_base_next, q_next)
        return x_q_next
