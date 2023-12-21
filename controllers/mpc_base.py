import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class MPC_Base:
    def __init__(self,
        base_demo, 
        obstacle1,
        N = 10, 
        Q = 2*np.diag([1., 2.]), 
        P = 2*np.diag([1., 1.]),
        R = np.diag([1]), 
        ulim=np.array([(0, -50),(2, 50)]), 
        xlim=np.array([(-100, -100),(100, 100)])
        ):

        self.Q = Q
        self.R = R
        self.P = P
        self.dt = base_demo.dt
        self.N = N
        self.ulim = ulim
        self.xlim = xlim
        self.f_dynamics = base_demo.f_kinematics #member function
        self.base_radius = base_demo.base_radius #member function
        self.obstacle1 = obstacle1
        # self.obstacle2 = obstacle2
        self.reset()
    def obs_avoid(self, obstacle, x):
        return obstacle.radius + self.base_radius()-ca.sqrt((x[0]-obstacle.x)**2 + (x[1]-obstacle.y)**2) #R_base+R_obstacle-|postion_base-postion_obstacle|<=0
        # return (x[0]-obstacle.x)**2 + (x[1]-obstacle.x)**2 - obstacle.radius - self.base_radius() 
    def reset(self):
        # Define optimization variables
        self.opti = ca.Opti()

        '''
        reason for the implementation that X is Nx2 not 2xN: 
        given p is a vector of same dim of x
        if we use Nx2, then matrix mult = X @ p; and elementwise mult = X * p, 
        of we use 2xN, then matrix mult = (X.T @ p).T; and elementwise mult = (X.T * p).T
        '''
        self.X = self.opti.variable(self.N+1, 6)    # states
        self.U = self.opti.variable(self.N, 2)      # inputs

        self.X_init = self.opti.parameter(1, 6)
        self.X_ref = self.opti.parameter(self.N+1, 2)
        self.U_ref = self.opti.parameter(self.N, 2)

        self.X_guess = None
        self.U_guess = None
        cost = 0
        # Define constraints and cost
        for k in range(self.N):
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))
            state_error = self.X[k, :2] - self.X_ref[k, :]
            control_error = self.U[k, :2] - self.U_ref[k, :]
            cost += ca.mtimes([state_error, self.Q, state_error.T]) \
                                + ca.mtimes([control_error, self.R, control_error.T])
            self.opti.subject_to(self.opti.bounded(self.ulim[0].reshape((1,2)), self.U[k, :], self.ulim[1].reshape((1,2)))) # control input constraint
            self.opti.subject_to(self.opti.bounded(self.xlim[0].reshape((1,2)), self.X[k, 0:2], self.xlim[1].reshape((1,2)))) # state constraint
            self.opti.subject_to(self.obs_avoid(self.obstacle1, self.X[k,:]) <=0) # obstacle1 avoidance
            # self.opti.subject_to(self.obs_avoid(self.obstacle2, self.X) <=0) # obstacle2 avoidance
            
        terminal_state_error = self.X[self.N, :2] - self.X_ref[self.N, :]
        print('terminal_state_error,',terminal_state_error.shape)
        cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])
        self.opti.subject_to(self.X[0, :] == self.X_init)# Initial state as constraints
        self.opti.subject_to(self.opti.bounded(self.xlim[0].reshape((1,2)), self.X[self.N, 0:2], self.xlim[1].reshape((1,2)))) # state constraint
        self.opti.subject_to(self.obs_avoid(self.obstacle1, self.X[self.N,:]) <=0)
        
        self.opti.minimize(cost)

        
        
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
            self.X_guess = np.ones((self.N+1, 6)) * x_init

        if self.U_guess is None:
            self.U_guess = np.zeros((self.N, 2))
        
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
            self.U_guess = self.U_guess.reshape(-1,2)
        return self.U_guess[0, :]
        


