import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class MPC_Base:
    def __init__(self,
        base_demo, 
        obstacle_list,
        N = 10, 
        Q = np.diag([2., 2., 0., 0, 0, 1.]), # x y psi dx dy dpsi
        P = np.diag([2., 2., 0., 0, 0, 1.]), 
        R = np.diag([1., 1.]), 
        M = np.diag([1e2]),
        ulim=np.array([[-2, -ca.pi],[2, ca.pi]]), # dv, dw
        xlim=np.array([[-100, -100, -2, -2, -ca.pi],[100, 100, 2, 2, ca.pi]]) # x, y, _, dx, dy, dpsi
        ):

        self.Q = Q
        self.R = R
        self.P = P
        self.M = M
        self.dt = base_demo.dt
        self.N = N
        self.ulim = ulim
        self.xlim = xlim
        self.f_dynamics = base_demo.f_kinematics #member function
        self.base_radius = base_demo.base_radius #member function
        self.obstacle_list = obstacle_list
        self.reset()

    def slackObsAvoid(self, obstacle_list, x):
        """
        not used!!! wrong implementation, slack variable should be explicitly used as the boundary of constraints, to avoid fmax() and fabs()
        returns the norm of the violation of constraints
        """
        expand_dist = 0.0
        sum_slack_var = 0
        for obs in obstacle_list:
            # dist_to_obs = ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) - (obs.radius + self.base_radius())
            # sum_slack_var += ca.fabs(ca.fmin(dist_to_obs - expand_dist, 0))
            dist_to_obs = ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) - (obs.radius + self.base_radius())
            # sum_slack_var += ca.fmax(- dist_to_obs + expand_dist, 0)
            sum_slack_var += - dist_to_obs + expand_dist
        return sum_slack_var
        #R_base+R_obstacle-|postion_base-postion_obstacle|<=0
        # return (x[0]-obstacle.x)**2 + (x[1]-obstacle.x)**2 - obstacle.radius - self.base_radius() 

    def obsAvoid(self, obstacle_list, x):
        g = []
        for obs in obstacle_list:
            g.append((obs.radius + self.base_radius()) - ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2)) # should be <= 0
        return g # all elements should be <= 0

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
        self.s = self.opti.variable(self.N+1, 1)    # slack variable

        self.X_init = self.opti.parameter(1, 6)
        self.X_ref = self.opti.parameter(self.N+1, 6)
        self.U_ref = self.opti.parameter(self.N, 2)

        self.X_guess = None
        self.U_guess = None
        cost = 0
        # Define constraints and cost, casadi requires x[k, :] instead of x[k] (which will be shape(1,1)) when calling row vector
        for k in range(self.N):
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))
            state_error = self.X[k, :] - self.X_ref[k, :]
            control_error = self.U[k, :] - self.U_ref[k, :]
            cost += ca.mtimes([state_error, self.Q, state_error.T]) \
                                + ca.mtimes([control_error, self.R, control_error.T])
            self.opti.subject_to(self.opti.bounded(self.ulim[0].reshape(1,2), self.U[k, :], self.ulim[1].reshape(1,2))) # control input constraint
            self.opti.subject_to(self.opti.bounded(self.xlim[0, 0:2].reshape(1,2), self.X[k, 0:2], self.xlim[1, 0:2].reshape(1,2))) # state constraint
            self.opti.subject_to(self.opti.bounded(self.xlim[0, 2:].reshape(1,-1), self.X[k, 3:], self.xlim[1, 2:].reshape(1,-1))) # state constraint
            for g in self.obsAvoid(self.obstacle_list, self.X[k,:]):
                self.opti.subject_to(g <= self.s[k])
            # constraint_error = self.slackObsAvoid(self.obstacle_list, self.X[k, :])
            cost += ca.mtimes([self.s[k], self.M, self.s[k]])
            
        terminal_state_error = self.X[self.N, :] - self.X_ref[self.N, :]
        print('terminal_state_error,',terminal_state_error.shape)
        cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])
        self.opti.subject_to(self.X[0, :] == self.X_init)# Initial state as constraints
        self.opti.subject_to(self.opti.bounded(self.xlim[0, 0:2].reshape(1,2), self.X[self.N, 0:2], self.xlim[1, 0:2].reshape(1,2))) # state constraint
        self.opti.subject_to(self.opti.bounded(self.xlim[0, 2:].reshape(1,-1), self.X[self.N, 3:], self.xlim[1, 2:].reshape(1,-1))) # state constraint
        for g in self.obsAvoid(self.obstacle_list, self.X[self.N,:]):
                self.opti.subject_to(g <= self.s[self.N])
        # self.opti.subject_to(self.obsAvoid(self.obstacle_list, self.X[self.N,:]) <= self.s[self.N])
        # terminal_constraint_error = self.slackObsAvoid(self.obstacle_list, self.X[self.N, :])
        cost += ca.mtimes([self.s[self.N], self.M, self.s[self.N]])
        
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
        self.opti.set_initial(self.s, np.zeros((self.N+1, 1)))

        self.opti.set_value(self.X_ref, traj_ref)
        self.opti.set_value(self.U_ref, u_ref)
        
        self.opti.set_value(self.X_init, x_init)

        try:
            sol = self.opti.solve()
            s = self.opti.debug.value(self.s)
        except:
            print("here should be a debug breakpoint")
            x = self.opti.debug.value(self.X)
            for x_k in x:
                print(self.obsAvoid(self.obstacle_list, x_k))
            print("x:", self.opti.debug.value(self.X))
            print("y:", self.opti.debug.value(self.U))
            print("s:", self.opti.debug.value(self.s))
        
        ## obtain the initial guess of solutions of the next optimization problem
        self.X_guess = sol.value(self.X)
        self.U_guess = sol.value(self.U) 
        if self.U_guess.ndim == 1: 
            self.U_guess = self.U_guess.reshape(-1,2)
        return self.U_guess[0, :]
        


