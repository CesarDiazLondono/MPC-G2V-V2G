"""
This script is used to evaluate the performance of the EVsSimulator environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ocmf_mpc import OCMF_V2G, OCMF_G2V
from eMPC import eMPC_V2G, eMPC_G2V


import numpy as np
import matplotlib.pyplot as plt

def eval():
    """
    Runs an evaluation of the EV2Gym environment.
    """
    save_plots = True
    
    replay_path = None

    config_file = "V2G_MPC.yaml"

    env = EV2Gym(config_file=config_file,
                       load_from_replay_path=replay_path,
                       verbose=True,
                       save_replay=True,                       
                       save_plots=save_plots,
                       )

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    ev_profiles = env.EVs_profiles    
    print(f'Number of EVs: {len(ev_profiles)}')
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival 
                            for ev in ev_profiles])
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    
    print(f'Max time of stay: {max_time_of_stay}')
    print(f'Min time of stay: {min_time_of_stay}')
    
    # agent = OCMF_V2G(env, control_horizon=25, verbose=False)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=False)
    # agent = eMPC_V2G(env, control_horizon=25, verbose=False)
    agent = eMPC_G2V(env, control_horizon=25, verbose=False)

    for t in range(env.simulation_length):        
        actions = agent.get_action(env)

        new_state, reward, done, _, stats = env.step(
            actions, visualize=True)  # takes action        

        if done:
            print(stats)
            print(f'End of simulation at step {env.current_step}')
            break


if __name__ == "__main__":
    eval()

