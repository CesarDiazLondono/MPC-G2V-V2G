"""
This script is used to evaluate the performance of the EVsSimulator environment.
"""
from EVsSimulator.ev_city import EVsSimulator
from EVsSimulator.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from EVsSimulator.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from occf_mpc import OCCF_V2G, OCCF_G2V
from eMPC import eMPC_V2G, eMPC_G2V

from EVsSimulator.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from EVsSimulator.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

import numpy as np
import matplotlib.pyplot as plt

def eval():
    """
    Runs an evaluation of the EVsSimulator environment.
    """
    save_plots = True
    
    replay_path = "./replay/replay_sim_2024_02_21_056441.pkl"
    replay_path = None

    config_file = "V2G_MPC.yaml"

    env = EVsSimulator(config_file=config_file,
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
    
    # agent = OCCF_V2G(env, control_horizon=25, verbose=False)
    # agent = OCCF_G2V(env, control_horizon=25, verbose=False)
    # agent = eMPC_V2G(env, control_horizon=25, verbose=False)
    # agent = V2GProfitMaxOracle(env,verbose=False)
    agent = eMPC_G2V(env, control_horizon=25, verbose=False)
    # agent = ChargeAsLateAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossible()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()    

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

