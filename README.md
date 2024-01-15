# RO47005 Planning and Decision Making
Group 34: Xinyu Gao, Weilin Xia, Pengzhi Yang, Jade Leurs, Jan 14, 2024
Project GitHub URL: https://github.com/HsinyuG/mobile-manipulator-mpc     
## 0. Task Description
- Autonomously navigate a mobile manipulator towards a designated target, specifically to push a button
- Avoid both static (and dynamic) obstacles in the environment

## 1. Installation
```angular2html
git clone https://github.com/HsinyuG/mobile-manipulator-mpc.git
cd mobile-manipulator-mpc
conda create -n mobile-manipulator python=3.9 -y
conda activate mobile-manipulator
pip install -r requirements.txt
```
Pybullet-based simulation environment [gym_envs_urdf](https://github.com/maxspahn/gym_envs_urdf) is used. Please install it before running the code.


## 2. Code Structure
    mobile-manipulator-mpc/
    ├── controllers/
    |         └── mpc_wholebody-qref.py  
    |                       └── Class MPCWholeBody
    |                                   ├── reset(...)
    |                                   ├── solve(...)
    |                                   └── ...
    ├── robot_models/
    |       ├── base.py
    |       |     └── Class Base 
    |       |             └── f_kinematics(...)  
    |       ├── manipulator_3DoF.py
    |       |           └──class ManipulatorPanda3DoF
    |       |                           ├── forward_tranformation(...)
    |       |                           └── inverse_transformation(...)
    |       ├── mobile_manipulator.py
    |       |           └──class MobileManipulator
    |       |                       ├── forward_tranformation(...)
    |       |                       └── f_kinematics(...)
    |       └── obstacles.py
    |               └── class Obstacles
    ├── simulation/
    |       ├── albert_robot.py
    |       |        └── ...   
    |       ├── obstacles.py 
    |       |        └── ...
    |       └── run_simulation.py 
    |                   └── ...
    ├── interface_wholebody_qref.py                   # Interface with simulation
    |             └── class Interface
    |                      ├── pseudoTimer(...) 
    |                      ├── observationCallback(...) 
    |                      ├── actuate(...)
    |                      ├── stateMachineUpdate(...)  
    |                      ├── timerCallback(...)
    |                      └── ...                 
    ├── demo_wholebody_qref.py                        # Main file to run the static obstacle scenario
    └── demo_wholebody_separate.py                    # Main file to run the dynamic obstacle scenario (it's in the 'moving_obs' branch)

## 3. Code Pointers
`controllers/mpc_wholebody_qref.py`:
This script implements MPC for the wholebody (mobile base + manipulator).
- `reset(self)`: set up the optimization variables, cost functions and constraints for the mobile manipulator
- `solve(self, x_init, traj_ref, u_ref)`: initialize and solve the optimization problem, handling potential errors and updating the initial guess for the next optimization iteration.
  
`interface_wholebody_qref.py`:
This script orchestrates the operation of the mobile manipulator, managing the simulation process and handoffing the control using a state machine. 
- `timerCallback(self)`: The main function in this script and the operational flow proceeds as follows:
  - add a synchronized timer with the simulation, managing the timing of simulation steps and MPC updates.
  - call observationCallback to get current state
  - calculate local reference trajectory
  - manage the state machine of the mobile maniulator, updating task flags based on current states and objectives, and check for task completion.
  - solve the mpc problem
  - apply commands to simulation, get observation
  
`demo_wholebody_qref.py`:
This script is the main file to run the project.
## 4. Run
###  experiment scenario of static obstacles
- `git checkout wholebody-qref` 
<!-- - `cd` to the directory of file: `mobile-manipulator-mpc/demo_wholebody_qref.py` -->
- select an experiment_scenario by changing the parameter `experiment_scenario` 
  - 0: for debug
  - 1: for obstacle avoidance during arm manipulating (avoid the corner of the table)
  - 2: for obstacle avoidance during base movement (avoid static obstacles both on the ground and above)
- run script `demo_wholebody_qref.py`
###  experiment scenario of dynamic obstacles  
- `git checkout moving_obs` 
<!-- - `cd` to the directory of file: `mobile-manipulator-mpc/demo_wholebody_separate.py` -->
- run script `demo_wholebody_separate.py`
## 5. Experimental Results
<div style="display:flex;justify-content: center;">
  <img src="./imgs/press_button.gif" width="60%">
</div>
<div style="display:flex;justify-content: center;">
  <img src="./imgs/aerial_obs.gif" width="60%">
</div>
<div style="display:flex;justify-content: center;">
  <img src="./imgs/moving_obs.gif" width="60%">
</div>

[//]: # (## 6. Contact)

[//]: # (Xinyu Gao - X.Gao-14@student.tudelft.nl)

[//]: # (Weilin Xia - W.Xia-3@student.tudelft.nl)

[//]: # (Pengzhi Yang - P.Yang-4@student.tudelft.nl)

[//]: # (Jade Leurs - J.Y.M.Leurs@student.tudelft.nl)

## 6. Contact
- Xinyu Gao - [X.Gao-14@student.tudelft.nl](mailto:X.Gao-14@student.tudelft.nl)
- Weilin Xia - [W.Xia-3@student.tudelft.nl](mailto:W.Xia-3@student.tudelft.nl)
- Pengzhi Yang - [P.Yang-4@student.tudelft.nl](mailto:P.Yang-4@student.tudelft.nl)
- Jade Leurs - [J.Y.M.Leurs@student.tudelft.nl](mailto:J.Y.M.Leurs@student.tudelft.nl)
