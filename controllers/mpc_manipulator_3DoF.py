import casadi as ca
import numpy as np
from casadi import *

M = 1e6

class MPCManipulator3DoF:
    def __init__(self,
        robot,
        N = 10,
        Q = np.diag([1, 1, 1]),  # q1 q2 q3
        P = np.diag([1, 1, 1]),
        R = np.diag([0, 0, 0]), # dq1 dq2 dq3 # 5e-8 both slow and jitter, 0 jitter
        M = np.diag([0, 0, 0]), # ddq1, ddq2, ddq3
        qlim=(ca.horzcat(-ca.pi/2, -ca.pi*3/4, 0), ca.horzcat(ca.pi/2, 0, ca.pi)),  # TODO: check the boundary
        dqlim=(ca.horzcat(-1, -1, -1), ca.horzcat(1, 1, 1)),
        ddqlim=(ca.horzcat(-0.5, -0.5, -0.5), ca.horzcat(0.5, 0.5, 0.5))):

        self.Q = Q
        self.R = R
        self.P = P
        self.M = M
        self.dt = robot.dt
        self.N = N
        self.qlim = qlim
        self.dqlim = dqlim
        self.ddqlim = ddqlim
        # System dynamics, which is a kinematic model
        self.f_dynamics = robot.f_kinematics
        self.robot_model = robot
        self.is_cartesian_ref = False # TODO: use as parameter, when true the reference is given in cartesian space
        self.normals = [np.array([[0, 0, -1]]), np.array([[1, 0, 0]])]
        self.obstacle_point = np.array([[0.25, 0, 0.3]])
        self.reset()

    def reset(self):
        # Define optimization variables
        self.opti = ca.Opti()
        '''
        reason for the implementation that X is Nx2 not 2xN: 
        given p is a vector of same dim of x
        if we use Nx2, then matrix mult = X @ p; and elementwise mult = X * p, 
        of we use 2xN, then matrix mult = (X.T @ p).T; and elementwise mult = (X.T * p).T
        '''
        self.X = self.opti.variable(self.N+1, 3)    # states are [q1 q2 q3].T
        self.U = self.opti.variable(self.N, 3)      # inputs are [dq1 dq2 dq3].T

        self.U_last = self.opti.parameter(self.N, 3)
        self.X_init = self.opti.parameter(1, 3)

        self.X_ref = self.opti.parameter(self.N+1, 3)
        self.U_ref = self.opti.parameter(self.N, 3)

        self.x_guess = None
        self.u_latest = None

        self.cost = 0
        # Define constraints and cost
        for k in range(self.N):
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))

            self.x_endpoint, self.x_joint_2, self.x_joint_3 = self.robot_model.forward_tranformation(self.X[k, :])
            if self.is_cartesian_ref:
                state_error = self.x_endpoint - self.X_ref[k, :]
            else: state_error = self.X[k, :] - self.X_ref[k, :]

            control_error = self.U[k, :] - self.U_ref[k, :]
            control_change = self.U[k, :] - self.U_last[k, :]

            self.cost += ca.mtimes([state_error, self.Q, state_error.T])
            self.cost += ca.mtimes([control_error, self.R, control_error.T])
            self.cost += ca.mtimes([control_change, self.M, control_change.T])

            self.opti.subject_to(self.opti.bounded(self.dqlim[0], self.U[k, :], self.dqlim[1])) # dq constraint
            self.opti.subject_to(self.opti.bounded(self.qlim[0], self.X[k, :], self.qlim[1])) # q constraint
            self.opti.subject_to(self.opti.bounded(self.ddqlim[0], control_change, self.ddqlim[1]))

            self.positions = [self.x_joint_2 / 2, self.x_joint_2, (self.x_joint_2 + self.x_joint_3) / 2, self.x_joint_3,
                              (self.x_joint_3 + self.x_endpoint) / 2, self.x_endpoint]

            self.obstacle_avoidance_constraints_convex()
            self.cost += self.slack**2 * M

        self.x_endpoint_terminal, self.x_joint_2_terminal, self.x_joint_3_terminal = \
            self.robot_model.forward_tranformation(
            self.X[self.N, :])
        if self.is_cartesian_ref:
            terminal_state_error = self.x_endpoint_terminal - self.X_ref[self.N, :]
        else: terminal_state_error = self.X[self.N, :] - self.X_ref[self.N, :]
        # TODO: xyz constraints
        self.cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])

        self.opti.minimize(self.cost)

        # Initial state as constraints
        self.opti.subject_to(self.X[0, :] == self.X_init)
        self.positions = [self.x_joint_2_terminal / 2, self.x_joint_2_terminal,
                                   (self.x_joint_2_terminal + self.x_joint_3_terminal) / 2, self.x_joint_3_terminal,
                          (self.x_joint_3_terminal + self.x_endpoint_terminal) / 2, self.x_endpoint_terminal]
        self.obstacle_avoidance_constraints_convex()
        self.cost += self.slack ** 2 * M

        # Set solver options
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}
        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, traj_ref, u_ref):

        # debug, disable the infeasible sensor feedback, TODO: slack variable
        x_init = np.maximum(np.minimum(x_init, self.qlim[1]), self.qlim[0]).squeeze()
        assert x_init[1] <= 0 and x_init[2] >= 0

        # Set initial guess for the optimization problem
        if self.x_guess is None:
            self.x_guess = np.ones((self.N+1, 3)) * x_init

        if self.u_latest is None:
            self.u_latest = np.zeros((self.N, 3))

        self.opti.set_initial(self.X, self.x_guess)
        self.opti.set_initial(self.U, self.u_latest)

        self.opti.set_value(self.X_ref, traj_ref)
        self.opti.set_value(self.U_ref, u_ref)
        # set the U_last for next solve() call
        self.opti.set_value(self.U_last, self.u_latest)

        self.opti.set_value(self.X_init, x_init)

        # sol = self.opti.solve()

        # Before the solve call
        self.opti.callback(lambda i: print("Iteration:", i, "Cost:", self.opti.debug.value(self.cost)))

        # Solve the problem
        try:
            sol = self.opti.solve()
            for i in range(len(self.positions)):
                print("Position", i, ":", self.opti.debug.value(self.positions[i]))
                print("Constraint 1:", self.opti.debug.value(self.constr[i, 0]))
                print("Constraint 2:", self.opti.debug.value(self.constr[i, 1]))
        except RuntimeError as e:
            print("Optimization failed:", str(e))
            # Inspect the values of variables and constraints at failure
            for i in range(len(self.positions)):
                print("Position", i, ":", self.opti.debug.value(self.positions[i]))
                print("Constraint 1:", self.opti.debug.value(self.constr[i, 0]))
                print("Constraint 2:", self.opti.debug.value(self.constr[i, 1]))
            raise
        print("\n")


        # obtain the initial guess of solutions of the next optimization problem
        self.x_guess = sol.value(self.X)
        self.u_latest = sol.value(self.U) # shape == (self.N, 3)

        # TODO: replace with u_latest
        return self.u_latest[0, :]

    def obstacle_avoidance_constraints_convex(self):
        # self.constr = self.opti.variable(1, 2)
        #
        # self.agent2obs = self.obstacle_point - self.positions[5]
        #
        # # Initialize the first constraint
        # self.constr[0, 0] = ca.mtimes(self.normals[0], self.agent2obs.T)
        # self.constr[0, 1] = ca.mtimes(self.normals[1], self.agent2obs.T)
        #
        # # Select the larger constraint using if_else
        # max_constr = ca.if_else(self.constr[0, 0] > self.constr[0, 1], self.constr[0, 0], self.constr[0, 1])
        #
        # # Add the maximum constraint to the optimization problem
        # self.opti.subject_to(self.constr[0, 1] > 0)

        self.constr = self.opti.variable(6, 2)
        self.slack = self.opti.variable()
        self.opti.subject_to(self.slack >= 0)

        for i in range(len(self.positions)):
            # Compute vector from position to obstacle point
            self.agent2obs = self.obstacle_point - self.positions[i]

            # Initialize the first constraint
            self.constr[i, 0] = ca.mtimes(self.normals[0], self.agent2obs.T)
            self.constr[i, 1] = ca.mtimes(self.normals[1], self.agent2obs.T)

            # Select the larger constraint using if_else
            max_constr = ca.if_else(self.constr[i, 0] > self.constr[i, 1], self.constr[i, 0], self.constr[i, 1])

            # Add the maximum constraint to the optimization problem
            self.opti.subject_to(max_constr > -self.slack)


