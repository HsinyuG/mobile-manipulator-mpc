import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

class MPC:
    def __init__(self,
        robot, 
        N = 10, 
        Q = np.diag([1., 0.0]), 
        P = np.diag([1., 0.0]), 
        R = np.diag([0.1]), 
        vlim=(-1, 1), 
        alim=(-5, 5)): 

        self.Q = Q
        self.R = R
        self.P = P
        self.dt = robot.dt
        self.N = N
        self.vlim = vlim
        self.alim = alim
        # System dynamics, which is a kinematic model
        self.f_dynamics = robot.f_kinematics

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
        self.X = self.opti.variable(self.N+1, 2)    # states
        self.U = self.opti.variable(self.N, 1)      # inputs

        self.X_init = self.opti.parameter(1, 2)

        self.X_ref = self.opti.parameter(self.N+1, 2)
        self.U_ref = self.opti.parameter(self.N, 1)

        self.X_guess = None
        self.U_guess = None
        
        cost = 0
        # Define constraints and cost
        for k in range(self.N):
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))
            state_error = self.X[k, :] - self.X_ref[k, :]
            control_error = self.U[k, :] - self.U_ref[k, :]
            cost += ca.mtimes([state_error, self.Q, state_error.T]) \
                                + ca.mtimes([control_error, self.R, control_error.T])
            self.opti.subject_to(self.opti.bounded(self.alim[0], self.U[k, :], self.alim[1])) # acc constraint
            self.opti.subject_to(self.opti.bounded(self.vlim[0], self.X[k, 1], self.vlim[1])) # vel constraint

        terminal_state_error = self.X[self.N, :] - self.X_ref[self.N, :]
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

        # Set initial guess for the optimization problem
        if self.X_guess is None:
            self.X_guess = np.ones((self.N+1, 2)) * x_init

        if self.U_guess is None:
            self.U_guess = np.zeros((self.N, 1))
        
        self.opti.set_initial(self.X, self.X_guess)
        self.opti.set_initial(self.U, self.U_guess)

        self.opti.set_value(self.X_ref, traj_ref)
        self.opti.set_value(self.U_ref, u_ref)
        
        self.opti.set_value(self.X_init, x_init)

        sol = self.opti.solve()
        
        ## obtain the initial guess of solutions of the next optimization problem
        self.X_guess = sol.value(self.X)
        self.U_guess = sol.value(self.U) 
        if self.U_guess.ndim == 1: 
            self.U_guess = self.U_guess.reshape(-1,1)
        return self.U_guess[0, :]
        


