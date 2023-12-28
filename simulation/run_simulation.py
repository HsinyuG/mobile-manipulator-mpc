import numpy as np

from albert_robot import (setup_environment, run_step)

def run(env, n_steps):
    action = np.zeros(env.n())
    action[0] = 0.2
    
    history = []

    for _ in range(n_steps):
        observation = run_step(env, action)
        history.append(observation)
    env.close()

    return history

if __name__ == "__main__":
    env = setup_environment(render=True, reconfigure_camera=True, obstacles=True, mode='vel')
    history = run(env, n_steps=1000)
    # print("history: ", history) # doesn't work for some reason