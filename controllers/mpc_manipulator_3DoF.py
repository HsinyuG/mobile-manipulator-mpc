import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

class MPCManipulator3DoF:
    def __init__(self,
        robot, 
        N = 10, 
        Q = np.diag([1, 0., 1]),  # q1 q2 q3
        P = np.diag([1, 0., 1]), 
        R = np.diag([0, 0, 0]), # dq1 dq2 dq3 # 5e-8 both slow and jitter, 0 jitter
        M = np.diag([1e-6, 1e-6, 1e-6]), # ddq1, ddq2, ddq3
        qlim=(ca.horzcat(-ca.pi/2, -ca.pi*3/4, 0), ca.horzcat(ca.pi/2, 0, ca.pi*3/2)),  # TODO: check the boundary
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
        
        cost = 0
        # Define constraints and cost
        for k in range(self.N):
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))

            if self.is_cartesian_ref:
                x_endpoint, x_joint_2, x_joint_3 = self.robot_model.forward_tranformation(self.X[k, :]) 
                state_error = x_endpoint - self.X_ref[k, :]
            else: state_error = self.X[k, :] - self.X_ref[k, :]
            # TODO: slack variable 

            control_error = self.U[k, :] - self.U_ref[k, :]
            control_change = self.U[k, :] - self.U_last[k, :]

            cost += ca.mtimes([state_error, self.Q, state_error.T])
            cost += ca.mtimes([control_error, self.R, control_error.T])
            cost += ca.mtimes([control_change, self.M, control_change.T])

            self.opti.subject_to(self.opti.bounded(self.dqlim[0], self.U[k, :], self.dqlim[1])) # dq constraint
            self.opti.subject_to(self.opti.bounded(self.qlim[0], self.X[k, :], self.qlim[1])) # q constraint
            self.opti.subject_to(self.opti.bounded(self.ddqlim[0], control_change, self.ddqlim[1]))
            
            # TODO: xyz constraints

        if self.is_cartesian_ref:
            x_endpoint_terminal, x_joint_2_terminal, x_joint_3_terminal = self.robot_model.forward_tranformation(self.X[self.N, :]) 
            terminal_state_error = x_endpoint_terminal - self.X_ref[self.N, :]
        else: terminal_state_error = self.X[self.N, :] - self.X_ref[self.N, :]
        # TODO: xyz constraints
        cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])

        self.opti.minimize(cost)

        # Initial state as constraints
        self.opti.subject_to(self.X[0, :] == self.X_init)

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

        sol = self.opti.solve()
        
        # obtain the initial guess of solutions of the next optimization problem
        self.x_guess = sol.value(self.X)
        self.u_latest = sol.value(self.U) # shape == (self.N, 3)

        # TODO: replace with u_latest
        return self.u_latest[0, :]
        


