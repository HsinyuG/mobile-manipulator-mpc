from robot_models import mobile_manipulator

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import copy

import simulation.albert_robot as sim

class Interface:
    def __init__(self, dt, t_move, t_manipulate, x_start, global_pose_target, controller, physical_sim=False):
        '''
        TODO: 
        1. assign values
        2. compute global reference trajectory 
        '''
        self.dt = dt
        self.desired_t_move = t_move
        self.desired_t_manipulate = t_manipulate
        self.global_pose_target = global_pose_target # x y z psi
        self.local_pose_target = None
        self.working_radius = 0.6
        self.x_target = np.array([
            global_pose_target[0] - self.working_radius * np.cos(global_pose_target[3]),
            global_pose_target[1] - self.working_radius * np.sin(global_pose_target[3]),
            global_pose_target[3],
            0, 0, 0,
            x_start[6], # q1
            x_start[7], # q2
            x_start[8]  # q3
        ])
        self.x_start = x_start
        self.controller = controller
        self.physical_sim = physical_sim

        self.manipulator_pose_log = []
        self.x_log = []
        self.u_log = []

        # counters and states
        self.sim_dt = 0.01
        self.timer_counter = 0
        self.mpc_step_counter = 0
        self.is_active = False

        if self.physical_sim:

            # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger]
            init_state = np.array([0.0, 0.0, 0.0, ca.pi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.idx_3dof = np.array([4,6,8])
            self.idx_base = np.array([0,1,2])
            init_state[self.idx_base] = self.x_start[:len(self.idx_base)]
            init_state[self.idx_3dof] = self.x_start[-len(self.idx_3dof):]
            self.env, self.ob = sim.setup_environment(render=True, reconfigure_camera=True, obstacles=True, mode='vel',\
                initial_state=init_state, dt=self.sim_dt) 
                
            self.vel_command_base = np.zeros(2) # V and omega, integrated by dV and dw command and used by simulation

        # self.globalPlan3D()
        self.traj_ref = None

    def run(self, ):
        '''
        TODO: 
        while(not end)
            1 call observationCallback to get current state (or its estimation)
            2 check if we finish the task or need to change the task
            3 calculate local reference for each optimization problem
            4 solve the mpc problem
            5 apply commands to simulation, get raw observation
            (6 calculate expected robot state if simulation part is not yet implemented)
        '''
        self.current_state = self.x_start if self.physical_sim is False else None # processed in observationCallback() in loop
        self.task_flag = 'move'
        self.is_active = True
        self.mpc_step_counter = 0
        while(self.is_active):
            self.pseudoTimer()

    
    def pseudoTimer(self):
        """ 
        This function acts as a timer, synchronized with the simulation time.
        No need to acquire sim time because gym won't update unless step() is called
        """
        if int(self.dt / self.sim_dt) <= 1:
            self.timerCallback()
            return
        if self.timer_counter == 0:
            self.timerCallback()
        else: self.actuate()
        self.timer_counter += 1
        if self.timer_counter == int(self.dt / self.sim_dt):
            self.timer_counter = 0

        
    def timerCallback(self):
        self.mpc_step_counter += 1
        print(self.mpc_step_counter, end=': ')

        # step 1
        if self.physical_sim is True: 
            self.observationCallback()
        print(self.task_flag)
        print(self.current_state)
        self.x_log.append(copy.deepcopy(self.current_state))
        self.current_joints_pose = np.asarray(ca.horzcat(*self.controller.robot_model.forward_tranformation(self.current_state))).squeeze()
        self.manipulator_pose_log.append(copy.deepcopy(self.current_joints_pose))

        # step 2
        # self.checkFinish3D()
        # if self.task_flag == 'finish': 
        #     self.is_active = False # can be calling another global planner
        #     return
        self.is_active = self.stateMachineUpdate()
        if not self.is_active: return

    
        # step 3, implemented in state machine
        # self.calcLocalRefTraj([0]) # 1d
        # self.calcLocalRefTraj([0,1,2]) # manipulator
        # self.calcLocalRefTraj([0,1,2]) # base, consider orientation error when finding corresponding point on global ref traj
        # self.calcLocalRefPose() # manipulator
        # self.calcLocalRefTraj([0,1,2], different_space=True) # wholebody, reference is in cartesian space

        # step 4
        self.command = self.controller.solve(self.current_state, self.local_traj_ref, self.local_u_ref)
        self.u_log.append(copy.deepcopy(np.asarray(self.command)))

        # step 5
        if self.physical_sim is True: 
            self.actuate()

        # step 6
        if self.physical_sim is False:
            self.current_state = np.asarray(self.controller.f_dynamics(self.current_state, self.command)).squeeze()


    def stateMachineUpdate(self):
        robot_status = True
        ## global path
        if self.task_flag == 'move' and self.traj_ref is None:
            self.globalPlan2D()

        ## calculate local ref traj and check move finish
        if self.task_flag == 'move' or self.task_flag == 'approach':

            if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= 2) and \
                (abs(self.current_state[1] - self.traj_ref[-1, 1]) <= 2) and \
                self.task_flag == 'move':
                # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
                # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
                self.task_flag = 'approach'
                # self.controller.setWeight(
                #     P = np.diag([5, 5, 5, 0, 0, 1.]), # 5e2 1
                #     Q = np.diag([5, 5, 5, 0, 0, 1.]),
                #     # R = np.diag([0, 0])
                # )
                N = self.controller.N
                self.controller.opti.subject_to(self.controller.X[N, :2] == self.controller.X_ref[N, :2])
            
            
            threshold = 0.2
            if (ca.norm_2(self.current_state[0:2] - self.traj_ref[-1, 0:2]) <= threshold):
                # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
                # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
                self.task_flag = 'rotate'
                self.controller.setWeight(
                    P = np.diag([5, 5, 5, 0, 0, 1, 1, 1, 1]),
                    Q = np.diag([5, 5, 5, 0, 0, 1, 1, 1, 1]),
                    # R = np.diag([0.1, 0.1, 0.0, 0.0, 0.0])

                    # P = np.diag([0, 0, 5, 0, 0, 0, 0, 0, 0]),
                    # Q = np.diag([0, 0, 5, 0, 0, 0, 0, 0, 0]),
                    # R = np.diag([0.0, 0.0, 0.0, 0.0, 0.0]),
                    # S = np.diag([0]),
                    # W = np.diag([0,0,0,0,0])
                )

            else: 
                self.calcLocalRefTraj([0,1]) if self.task_flag == 'move' else self.calcLocalRefPose()

        ## check rotate finish
        # TODO: set weight large dV penalty and V penalty, large psi penalty
        if self.task_flag == 'rotate':
            threshold = 0.01
            if (abs(self.controller.angleDiff(self.current_state[2], self.traj_ref[-1, 2])) <= 0.5*ca.pi/180) and \
                (ca.norm_2(self.current_state[0:2] - self.traj_ref[-1, 0:2]) <= threshold): 
                self.task_flag = 'move finish' # actuate function will send 0 vel; actual may perform better because we cannot brake in sim
            else: self.calcLocalRefPose()

        ## adjust final location
        # if self.task_flag == 'adjust':


        ## when move finish, change reference q value
        if self.task_flag == 'move finish':# and np.isclose(self.current_state[3:6], 0, atol=1e-04).all():
            self.task_flag = 'manipulate'
            self.local_pose_target = np.array([
                np.sqrt((self.global_pose_target[0] - self.current_state[0])**2 + (self.global_pose_target[1] - self.current_state[1])**2) + 0.007, # x difference between base link and joint 1
                0.0, # no DoF here in our manipulator
                self.global_pose_target[2] - (0.606+0.333) # height difference between /base_link and joint 1
            ])
            self.globalPlanManipulator() 
            self.controller.setWeight(
                P = np.diag([500, 500, 500, 0, 0, 1, 1, 1, 1]),
                Q = np.diag([500, 500, 500, 0, 0, 1, 1, 1, 1]),
                # R = np.diag([0.1, 0.1, 0.0, 0.0, 0.0])
            )

        ## check manipulate finish
        if self.task_flag == "manipulate":
            threshold = 0.01
            # maybe need to use x y z psi of endpoint as reference to move the base
            if (ca.norm_2(self.current_joints_pose[:3] - self.global_pose_target[:3]) <= threshold): # when psi goal=the psi it finish move, <= 0.07
                self.task_flag = 'manipulate finish'
                robot_status = False
            else:
                self.calcLocalRefTraj([6,7,8])
        
        return robot_status


    def globalPlan1D(self):
        '''
        TODO: 
        compute global reference trajectory 
        '''
        traj_length = int(self.desired_t_total/self.dt)

        # only give the positions, velocity references are not used thus will not be panalized
        self.traj_ref = np.array([
            np.linspace(self.x_start[0], self.pose_target[0], int(traj_length + 1)),
            np.zeros(int(traj_length + 1))
        ]).T

        self.u_ref = np.zeros(traj_length, 1)

    
    def globalPlan2D(self):
        '''
        TODO: 
        compute global reference trajectory 
        '''
        traj_length = int(self.desired_t_move/self.dt)

        # only give the positions, linear velocity references are not given thus will not be panalized
        # self.traj_ref = np.array([
        #     np.linspace(self.x_start[0], self.x_target[0], int(traj_length + 1)),
        #     np.linspace(self.x_start[1], self.x_target[1], int(traj_length + 1)),
        #     np.zeros(int(traj_length + 1)),
        #     np.zeros(int(traj_length + 1)),
        #     np.zeros(int(traj_length + 1)),
        #     np.zeros(int(traj_length + 1))
        # ]).T

        self.traj_ref = np.linspace(self.x_start, self.x_target, int(traj_length + 1))

        self.u_ref = np.zeros((traj_length, 5))


    def globalPlanManipulator(self):
        '''
        x_start: joint angles
        pose_target: endpoint pose
        '''
        traj_length = int(self.desired_t_manipulate/self.dt)
        ''' pose_start, _, _ = np.asarray(self.controller.robot_model.forward_tranformation(self.x_start)).squeeze()
        self.traj_ref = np.array([
            np.linspace(pose_start[0], self.pose_target[0], int(traj_length + 1)),
            np.linspace(pose_start[1], self.pose_target[1], int(traj_length + 1)),
            np.linspace(pose_start[2], self.pose_target[2], int(traj_length + 1)),
        ]).T

        not working because linear reference trajectory in cartesian space cause infeasibility'''

        x_target = np.hstack((
            self.current_state[:6],
            self.controller.robot_model.manipulator.inverse_transformation(self.current_state[-3:], self.local_pose_target)
        ))
        # self.traj_ref = np.array([
        #     np.linspace(self.x_start[0], x_target[0], int(traj_length + 1)),
        #     np.linspace(self.x_start[1], x_target[1], int(traj_length + 1)),
        #     np.linspace(self.x_start[2], x_target[2], int(traj_length + 1)),
        # ]).T
        self.traj_ref = np.linspace(self.current_state, x_target, int(traj_length + 1))

        self.u_ref = np.zeros((traj_length, 5))

    def globalPlan3D(self):
        traj_length = int(self.desired_t_total/self.dt)
        pose_start = np.asarray(self.controller.robot_model.forward_tranformation(self.x_start)[0]).squeeze()
        
        # x, y, z, psi of end point
        self.traj_ref = np.array([
            np.linspace(pose_start[0], self.pose_target[0], int(traj_length + 1)),
            np.linspace(pose_start[1], self.pose_target[1], int(traj_length + 1)),
            np.linspace(pose_start[2], self.pose_target[2], int(traj_length + 1)),
            np.linspace(pose_start[3], self.pose_target[3], int(traj_length + 1)),
        ]).T

        self.u_ref = np.zeros((traj_length,5))


    def checkFinish1D(self):
        '''
        check if we reach the goal
        '''
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= 0.5) and \
            (abs(self.current_state[1] - self.u_ref[-1, 0]) <= 1e-2):
            self.task_flag = 'finish'


    def checkFinish2D(self):
        '''
        check if we reach the goal
        '''
        threshold = 0.1
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= threshold) and \
            (abs(self.current_state[1] - self.traj_ref[-1, 1]) <= threshold): 
            # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
            # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
            self.task_flag = 'finish'


    def checkFinishManipulator(self):
        '''
        check if manipulator reach the target pose
        '''
        if (ca.norm_2(self.current_joints_pose[:3] - self.local_pose_target) <= 0.02):
            self.task_flag = 'finish'

    def checkFinish3D(self):
        if self.task_flag == 'move':
            self.controller.R = np.diag([0.1, 0.1, 1e2, 1e2, 1e2])
        if (ca.norm_2(self.current_joints_pose[:4] - self.pose_target) <= 1) and self.task_flag == 'move': 
            # self.controller.opti.subject_to(self.controller.terminal_pose_endpoint == self.controller.X_ref[self.controller.N, :])
            # self.controller.Q = 1e5*np.diag([1, 1, 1, 1])
            # self.controller.P = 1e5*np.diag([1, 1, 1, 1])

            self.task_flag = 'approach'
        
        if (ca.norm_2(self.current_joints_pose[:4] - self.pose_target) <= 0.1): # TODO: change W to make it 0.02
            self.task_flag = 'finish'

    def calcLocalRefTraj(self, distance_index, different_space=False):
        '''
        input:
        - x_ref: global trajectory given in cartesian space
        - distance_index: list, the indices of distance variable in the state vector
        - different_space: if states are in joint space but reference in are in cartesian space
        description: 
        - compute global reference trajectory 
        - if the goal lies within the horizon, repeat the last reference point
        '''
        distance_index = np.asarray(distance_index)
        min_distance = 1e5
        min_idx = -1e5
        for i, x_ref in enumerate(self.traj_ref):

            if different_space:
                distance = np.linalg.norm(self.current_joints_pose[:3] - x_ref[distance_index])
            else:
                distance = np.linalg.norm(self.current_state[distance_index] - x_ref[distance_index])

            if distance < min_distance:
                min_distance = distance
                min_idx = i

        terminal_index = min_idx + self.controller.N + 1   # reference (N+1) states and N inputs    
        if terminal_index <= self.traj_ref.shape[0]:
            self.local_traj_ref = self.traj_ref[min_idx : terminal_index]
            self.local_u_ref = self.u_ref[min_idx : terminal_index-1]
        else:
            # print(self.local_traj_ref.shape)
            # print(self.local_u_ref.shape)

            last_traj_ref = self.traj_ref[-1]
            last_u_ref = self.u_ref[-1]
            repeat_times = terminal_index - self.traj_ref.shape[0]
            self.local_traj_ref = np.vstack([self.traj_ref[min_idx :], np.tile(last_traj_ref, (repeat_times, 1))])
            self.local_u_ref = np.vstack([self.u_ref[min_idx :], np.tile(last_u_ref, (repeat_times, 1))])

            # print(self.local_traj_ref.shape)
            # print(self.local_u_ref.shape)
            # print(self.local_traj_ref)
        
        assert self.local_traj_ref.shape[0] == self.controller.N + 1
        assert self.local_u_ref.shape[0] == self.controller.N

    def calcLocalRefPose(self):
        # self.local_traj_ref = np.tile(self.pose_target, (self.controller.N + 1, 1))
        self.local_traj_ref = np.tile(self.traj_ref[-1], (self.controller.N + 1, 1))
        self.local_u_ref = np.tile(self.u_ref[-1], (self.controller.N, 1))

        """ 
        to avoid angle ref is -3.14 and current angle is 3.14, in this case we can use angle
        diff function in solver, but this makes it too difficult to solve
        """
        psi_ref = self.traj_ref[-1, 2]
        current_psi = self.current_state[2]
        continuous_psi_ref = current_psi + self.controller.angleDiff(psi_ref, current_psi)
        self.local_traj_ref[:, 2] = continuous_psi_ref


    def observationCallback(self):
        '''
        TODO: get the state feedback in simulation, 
        e.g. 
        self.observation = some function from simulation
        self.current_state = somehow(self.observation)
        '''
        # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger]

        if self.current_state is None:
            self.current_state = np.hstack([
                self.ob[0]['robot_0']['joint_state']['position'][self.idx_base],
                self.ob[0]['robot_0']['joint_state']['velocity'][self.idx_base],
                self.ob[0]['robot_0']['joint_state']['position'][self.idx_3dof]
            ])
        else: self.current_state = np.hstack([
                self.ob['robot_0']['joint_state']['position'][self.idx_base],
                self.ob['robot_0']['joint_state']['velocity'][self.idx_base],
                self.ob['robot_0']['joint_state']['position'][self.idx_3dof]
            ])

        # real_psi = self.current_state[2] + ca.pi
        # self.current_state[2] = ca.fmod((real_psi + ca.pi), (2*ca.pi)) - ca.pi # convert to [-pi, pi)
        self.current_state[3] *= -1
        self.current_state[4] *= -1

        offset = 0.157 # base center is (0.157, 0, 0) with reference to geometric center location which is in self.ob 
        self.current_state[0] += offset * ca.cos(self.current_state[2])
        self.current_state[1] += offset * ca.sin(self.current_state[2])


    def actuateBase(self):
        '''
        TODO: use the return of mpc.solve() to set commands in simulation
        e.g. 
        some function from simulation(self.command)
        '''
        if not self.physical_sim: return
        action = np.zeros(self.env.n())
        # shape=(11,), [v, w, joint1~7, left finger, right finger]
        idx_3dof = np.array([3,5,7]) # different from the self.idx_3dof, which is in state and observation
        idx_base = np.array([0,1]) # different from the self.idx_base, which is in state and observation
        self.vel_command_base += self.sim_dt * self.command
        action[idx_base] = self.vel_command_base
        self.ob = sim.run_step(self.env, action)

    def actuate3DoFManipulator(self, command=None):
        if not self.physical_sim: return
        action = np.zeros(self.env.n())
        # shape=(11,), [v, w, joint1~7, left finger, right finger]
        idx_3dof = np.array([3,5,7])
        action[idx_3dof] = self.command
        self.ob = sim.run_step(self.env, action)

    def actuate(self):
        if not self.physical_sim: return
        action = np.zeros(self.env.n())
        # shape=(11,), [v, w, joint1~7, left finger, right finger]
        idx_3dof = np.array([3,5,7]) # different from the self.idx_3dof, which is in state and observation
        idx_base = np.array([0,1]) # different from the self.idx_base, which is in state and observation
        self.vel_command_base += self.sim_dt * self.command[0:2] * np.array([-1,1]) # V's direction is reversed in sim
        # if self.task_flag in ('move', 'rotate', 'approach'): 
        action[idx_base] = self.vel_command_base
        # else: brake
        action[idx_3dof] = self.command[2:]
        self.ob = sim.run_step(self.env, action)

    def plot1D(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        plt.subplot(411)
        plt.plot(t, self.x_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('p')
        plt.grid()

        plt.subplot(412)
        plt.plot(t, self.x_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('v')
        plt.grid()

        plt.subplot(413)
        plt.plot(t[:-1], self.u_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('a')
        plt.grid()

        plt.subplot(414)
        plt.plot(t[:self.traj_ref.shape[0]], self.traj_ref[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('p ref')
        plt.grid()        

        plt.show()

    def plotManipulator(self, is_mobile=False):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        self.manipulator_pose_log = np.asanyarray(self.manipulator_pose_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        idx_offset = 6 if is_mobile else 0
        # q1 q2 q3
        plt.figure()
        plt.subplot(121)
        plt.plot(t, self.x_log[:, 0+idx_offset], label='q1')
        plt.plot(t, self.x_log[:, 1+idx_offset], label='q2')
        plt.plot(t, self.x_log[:, 2+idx_offset], label='q3')
        plt.xlabel('Time Step')
        plt.ylabel('joint angles')
        plt.legend()
        plt.grid()

        idx_offset = 2 if is_mobile else 0
        plt.subplot(122)
        plt.plot(t[:-1], self.u_log[:, 0+idx_offset], label='dq1')
        plt.plot(t[:-1], self.u_log[:, 1+idx_offset], label='dq2')
        plt.plot(t[:-1], self.u_log[:, 2+idx_offset], label='dq3')
        plt.xlabel('Time Step')
        plt.ylabel('joint velocities')
        plt.legend()
        plt.grid()

        idx_offset = 1 if is_mobile else 0
        # x y z
        plt.figure()   
        plt.subplot(131)
        plt.plot(t, self.manipulator_pose_log[:, 0], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 1], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 2], label='z')
        if is_mobile: plt.plot(t, self.manipulator_pose_log[:, 3], label='psi')
        plt.xlabel('Time Step')
        plt.ylabel('end point')
        plt.legend()
        plt.grid()

        plt.subplot(132)
        plt.plot(t, self.manipulator_pose_log[:, 3+idx_offset], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 4+idx_offset], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 5+idx_offset], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('joint 2')
        plt.legend()
        plt.grid()       

        plt.subplot(133)
        plt.plot(t, self.manipulator_pose_log[:, 6+idx_offset], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 7+idx_offset], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 8+idx_offset], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('joint 3')
        plt.legend()
        plt.grid()   

        plt.show(block=True)


    def plot2D(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        # x y psi
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
        plt.plot(t, self.x_log[:, 2])
        plt.xlabel('Time Step')
        plt.ylabel('psi')
        plt.grid()

        # dx, dy, dpsi
        plt.figure()
        plt.subplot(311)
        plt.plot(t, self.x_log[:, 3])
        plt.xlabel('Time Step')
        plt.ylabel('dx')
        plt.grid()

        plt.subplot(312)
        plt.plot(t, self.x_log[:, 4])
        plt.xlabel('Time Step')
        plt.ylabel('dy')
        plt.grid()

        plt.subplot(313)
        plt.plot(t, self.x_log[:, 5])
        plt.xlabel('Time Step')
        plt.ylabel('dpsi')
        plt.grid()
        
        # dV, dw
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
        
        # obs
        plt.figure()
        plt.plot(self.x_log[:, 0], self.x_log[:, 1],label = 'actual postion')
        ref_line = np.linspace(self.x_start[:2], self.x_target[:2])
        plt.plot(ref_line[:, 0], ref_line[:, 1], label = 'reference position')
        for obs in self.controller.obstacle_list:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='green', fill=False)
            plt.gca().add_artist(circle)
            plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show(block=False)


    def plot3D(self):
        self.plot2D()
        self.plotManipulator(is_mobile=True)
