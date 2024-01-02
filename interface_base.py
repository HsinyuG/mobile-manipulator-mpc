import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import math

class Interface:
    def __init__(self, dt, t_total, x_start, x_target, controller, physical_sim=False):
        '''
        TODO: 
        1. assign values
        2. compute global reference trajectory 
        '''
        self.dt = dt
        self.desired_t_total = t_total
        self.x_target = x_target
        self.x_start = x_start
        self.controller = controller
        self.physical_sim = physical_sim

        self.x_log = []
        self.u_log = []

        self.globalPlan2D()

    def run(self, ):
        '''
        TODO: 
        while(not end)
            1 call observationCallback to get state feedback
            2 check if we finish the task or need to change the task
            3 calculate local reference for each optimization problem
            4 solve the mpc problem
            5 apply commands to simulation
            (6 calculate expected robot state if simulation part is not yet implemented)
        '''
        self.current_state = self.x_start
        self.task_flag = 'in progress'
        num_step = 0
        while(True):
            num_step += 1
            print(num_step, end=': ')
            print(self.current_state)
            # step 1
            if self.physical_sim is True: 
                self.observationCallback()
            # TODO: get current state from simulation
            self.x_log.append(self.current_state)

            # step 2
            self.checkFinish2D()
            if self.task_flag != 'in progress': break # can be calling another global planner

            # step 3
            self.calcLocalRef()

            # step 4
            self.command = self.controller.solve(self.current_state, self.local_traj_ref, self.local_u_ref)
            self.u_log.append(np.asarray(self.command))

            # step 5
            if self.physical_sim is True: 
                self.actuate()

            # step 6
            if self.physical_sim is False:
                self.current_state = np.asarray(self.controller.f_dynamics(self.current_state, self.command)).squeeze()

    def globalPlan2D(self):
        '''
        TODO: 
        compute global reference trajectory 
        '''
        traj_length = int(self.desired_t_total/self.dt)

        # only give the positions, velocity references are not used thus will not be panalized
        self.traj_ref = np.array([np.linspace(self.x_start[0], self.x_target[0], int(traj_length + 1)),
                                 np.linspace(self.x_start[1], self.x_target[1], int(traj_length + 1))]
                                ).T

        self.u_ref = np.zeros((traj_length,2))

    def checkFinish2D(self):
        '''
        check if we reach the goal
        '''
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= 0.5) and \
            (abs(self.current_state[1] - self.traj_ref[-1, 1]) <= 0.5): 
            # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
            # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
            self.task_flag = 'finish'

    def calcLocalRef(self):
        '''
        TODO: 
        compute global reference trajectory 
        if the goal lies within the horizon, repeat the last reference point
        '''
        min_distance = 1e5
        min_idx = -1e5
        # choose the cloest traj_ref to be reference point
        for i, ref_point in enumerate(self.traj_ref):
            # print('ref_point',ref_point)
            distance = np.linalg.norm(self.current_state[:2] - ref_point)
            if distance < min_distance:
                min_distance = distance
                min_idx = i

        terminal_index = min_idx + self.controller.N + 1   # reference (N+1) states and N inputs    
        if terminal_index <= self.traj_ref.shape[0]:
            self.local_traj_ref = self.traj_ref[min_idx : terminal_index,:]
            self.local_u_ref = self.u_ref[min_idx : terminal_index-1,:] 
        else:
            # print(self.local_traj_ref.shape)
            # print(self.local_u_ref.shape)

            last_traj_ref = self.traj_ref[-1,:]
            # print(last_traj_ref)
            last_u_ref = self.u_ref[-1,:]
            repeat_times = terminal_index - self.traj_ref.shape[0]
            self.local_traj_ref = np.vstack([self.traj_ref[min_idx :, :], np.tile(last_traj_ref, (repeat_times, 1))])
            self.local_u_ref = np.vstack([self.u_ref[min_idx :, :], np.tile(last_u_ref, (repeat_times, 1))])

        # print('local_traj_ref', self.local_traj_ref.shape)
        # print('local_u_ref', self.local_u_ref.shape)
        assert self.local_traj_ref.shape[0] == self.controller.N + 1
        assert self.local_u_ref.shape[0] == self.controller.N

    def observationCallback(self):
        '''
        TODO: get the state feedback in simulation, 
        e.g. 
        self.observation = some function from simulation
        self.current_state = somehow(self.observation)
        '''
        pass

    def actuate(self):
        '''
        TODO: use the return of mpc.solve() to set commands in simulation
        e.g. 
        some function from simulation(self.command)
        '''
        pass

    def plot(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        # Plot the results
        t = np.arange(len(self.x_log))
        plt.figure()
        plt.subplot(311)
        plt.plot(t, self.x_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('x')
        plt.grid()

        plt.subplot(312)
        plt.plot(t, self.x_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('y')
        plt.grid()

        plt.subplot(313)
        plt.plot(t, self.x_log[:, 3])
        plt.xlabel('Time Step')
        plt.ylabel('psi')
        plt.grid()
        
        plt.figure()
        plt.subplot(211)
        plt.plot(t[:-1], self.u_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('v_dot')
        plt.grid()

        plt.subplot(212)
        plt.plot(t[:-1], self.u_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('w_dot')
        plt.grid()
        
        plt.figure()
        plt.plot(self.x_log[:, 0], self.x_log[:, 1],label = 'actual postion')
        plt.plot(self.traj_ref[:, 0], self.traj_ref[:, 1],label = 'reference position')
        circle = plt.Circle((25, 30), 4, color='green', fill=False)
        plt.gca().add_artist(circle)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()
