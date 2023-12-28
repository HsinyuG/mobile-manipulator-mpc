import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

class ManipulatorPanda3DoF:
    def __init__(self, dt):
        self.dt = dt
        # self.model_params = whatever, not used in this demo

    def forward_tranformation(self, q):
        """
        x: [x, y, z], may consider pitch later
        q: [q1, q2, q3]"""
        a2 = 0.316
        a3 = 0.0825
        a5 = 0.384
        a6 = 0.088
        a7 = 0.107

        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        # joint2
        T_joint2_to_base = ca.vertcat(
            ca.horzcat(ca.cos(q1), ca.sin(q1), 0, a2*ca.sin(q1) + a3*ca.cos(q1)),
            ca.horzcat(0, 0, -1, 0),
            ca.horzcat(-ca.sin(q1), ca.cos(q1), 0, a2*ca.cos(q1) - a3*ca.sin(q1)),
            ca.horzcat(0, 0, 0, 1)
        )
        R_base_to_joint2 = T_joint2_to_base[0:3, 0:3].T # the rotation is inversed, equivalent to transpose
        x_joint_2 = T_joint2_to_base[0:3, 3].T # row vector
        
        # joint 3
        T_joint3_to_base = ca.vertcat(
            ca.horzcat(
                ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1), 
                -ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2), 
                0, 
                a2*ca.sin(q1) - a3*ca.sin(q1)*ca.sin(q2) - a3*ca.cos(q1)*ca.cos(q2) + a3*ca.cos(q1) + a5*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))
            ), 
            ca.horzcat(0, 0, -1, 0), 
            ca.horzcat(
                ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2), 
                ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1), 
                0, 
                a2*ca.cos(q1) + a3*ca.sin(q1)*ca.cos(q2) - a3*ca.sin(q1) - a3*ca.sin(q2)*ca.cos(q1) + a5*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))
            ), 
            ca.horzcat(0, 0, 0, 1)
        )
        R_base_to_joint3 = T_joint3_to_base[0:3, 0:3].T
        x_joint_3 = T_joint3_to_base[0:3, 3].T

        T_endpoint_to_base = ca.vertcat(
            ca.horzcat(
                -(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) - (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3), 
                0, 
                -(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3), 
                a2*ca.sin(q1) - a3*ca.sin(q1)*ca.sin(q2) - a3*ca.cos(q1)*ca.cos(q2) + a3*ca.cos(q1) + a5*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1)) - a6*(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3) - a7*((-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3))
            ), 
            ca.horzcat(0, 1, 0, 0), 
            ca.horzcat(
                -(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) - (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3), 
                0, 
                (ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) - (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3), 
                a2*ca.cos(q1) + a3*ca.sin(q1)*ca.cos(q2) - a3*ca.sin(q1) - a3*ca.sin(q2)*ca.cos(q1) + a5*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) + a6*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) - a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3) - a7*((ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3))
            ),
            ca.horzcat(0, 0, 0, 1)
        )
        R_base_to_endpoint = T_endpoint_to_base[0:3, 0:3].T
        x_endpoint = T_endpoint_to_base[0:3, 3].T

        return x_endpoint, x_joint_2, x_joint_3    

    def _get_xdot(self, q, q_dot):
        """
        x: [x, y, z], may consider pitch later
        q: [q1, q2, q3]"""

        # DH table in Dynamic Identification of the Franka Emika Panda Robot With Retrieval of Feasible Parameters Using Penalty-Based Optimization 
        a2 = 0.316
        a3 = 0.0825
        a5 = 0.384
        a6 = 0.088
        a7 = 0.107

        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        dq1 = q_dot[0]
        dq2 = q_dot[1]
        dq3 = q_dot[2]

        # x_dot
        dx1 = dq1 * (a2*ca.cos(q1) + a3*ca.sin(q1)*ca.cos(q2) - a3*ca.sin(q1) - a3*ca.sin(q2)*ca.cos(q1) + a5*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) + a6*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) - a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3) - a7*((ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3))) \
            + dq2 * (-a3*ca.sin(q1)*ca.cos(q2) + a3*ca.sin(q2)*ca.cos(q1) - a5*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2)) - a6*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3) + a7*((ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3))) \
            + dq3 * (-a6*(ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3) + a7*((ca.sin(q1)*ca.sin(q2) + ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3)))
        
        # y_dot
        dx2 = dq1 * 0 \
            + dq2 * 0 \
            + dq3 * 0 

        # z_dot
        dx3 = dq1 * (-a2*ca.sin(q1) + a3*ca.sin(q1)*ca.sin(q2) + a3*ca.cos(q1)*ca.cos(q2) - a3*ca.cos(q1) - a5*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1)) + a6*(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) - a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3) + a7*((-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3))) \
            + dq2 * (-a3*ca.sin(q1)*ca.sin(q2) - a3*ca.cos(q1)*ca.cos(q2) + a5*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1)) - a6*(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3) - a7*((-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3))) \
            + dq3 * (-a6*(-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.cos(q3) + a6*(ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.sin(q3) - a7*((-ca.sin(q1)*ca.sin(q2) - ca.cos(q1)*ca.cos(q2))*ca.sin(q3) + (ca.sin(q1)*ca.cos(q2) - ca.sin(q2)*ca.cos(q1))*ca.cos(q3)))
            
        # roll_dot
        dx4 = dq1 * 0 \
            + dq2 * 0 \
            + dq3 * 0

        # pitch_dot
        dx5 = dq1 * (1) \
            + dq2 * (-1) \
            + dq3 * (-1)

        # yaw_dot
        dx6 = dq1 * 0 \
            + dq2 * 0 \
            + dq3 * 0        
        # print(dx1)
        # print(dx2)
        # print(dx3)
        return ca.vertcat(dx1, dx2, dx3)

    def f_kinematics(self, q, q_dot):
        q += q_dot * self.dt
        return q

    def f_kinematics_with_jacobian(self, x, q, q_dot):
        # TODO: assert x.type == u.type == casadi variable
        # x1 = x[0]
        # return ca.sin(x1)+ca.cos(q_dot)+q
        x_next = x + self._get_xdot(q, q_dot) * self.dt
        q_next = ca.vertcat(
            q[0] + q_dot[0] * self.dt,
            q[1] + q_dot[1] * self.dt,
            q[2] + q_dot[2] * self.dt
        )

        x_q_next = ca.vertcat(x_next, q_next)
        return x_q_next

# # debug
# toy = ManipulatorPanda3DoF(0.1)
# print(toy.f_kinematics(
#     # ca.vertcat(0, 0, 0), # x
#     ca.vertcat(0, 0, 0), # q
#     ca.vertcat(0.0, 0.0, 0) # dq
# ))

# print(toy.forward_tranformation(
#     ca.vertcat(0, 0, ca.pi/2)
# )[0].shape)