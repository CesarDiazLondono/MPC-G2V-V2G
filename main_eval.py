"""
This script is used to evaluate the performance of the EVsSimulator environment.
"""

import EVsSimulator
from EVsSimulator.ev_city import EVsSimulator
from EVsSimulator.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible

from mpc import MPC
import numpy as np
import matplotlib.pyplot as plt


def eval():
    """
    Runs an evaluation of the EVsSimulator environment.
    """

    verbose = False
    save_plots = True

    # replay_path = "./replay/replay_sim_2024_02_14_272496.pkl"
    replay_path = None

    config_file = "/v2g_config.yaml"    

    env = EVsSimulator(config_file=config_file,
                       load_from_replay_path=replay_path,
                       verbose=True,
                       save_replay=True,                       
                       save_plots=save_plots,                       
                       )

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    mpc = MPC(env, control_horizon=25, verbose=True)
    round_robin = RoundRobin(env, verbose=True)
    charge_as_late_as_possible = ChargeAsLateAsPossible(verbose=True)
    charge_as_fast_as_possible = ChargeAsFastAsPossible()
    rewards = []

    for t in range(env.simulation_length):
        # all ports are charging instantly        
        # actions = charge_as_fast_as_possible.get_action(env)
        actions = round_robin.get_action(env)
        # actions = charge_as_late_as_possible.get_action(env)
        # input("Press Enter to continue...")
        # MPC
        # actions = mpc.get_actions_OCCF(t=t)
        # actions = mpc.get_actions_economicV2G(t=t)
        # actions = mpc.get_actions_OCCF_with_Loads(t=t)        
        
        if verbose:
            print(f'Actions: {actions}')

        new_state, reward, done, truncated, _ = env.step(
            actions, visualize=True)  # takes action
        rewards.append(reward)

        # input("Press Enter to continue...")

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            print(f'End of simulation at step {env.current_step}')
            break

    env = EVsSimulator(config_file=config_file,
                       load_from_replay_path=new_replay_path,
                       verbose=True,
                       save_plots=True,
                       )
    state, _ = env.reset()
    rewards_opt = []

    for t in range(env.simulation_length):

        if verbose:
            print(f' OptimalActions: {actions}')

        new_state, reward, done, truncated, _ = env.step(
            actions, visualize=True)  # takes action
        rewards_opt.append(reward)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break

    if save_plots:
        plt.figure(figsize=(10, 10))
        # Plot the commulative reward in subplot 1
        plt.subplot(2, 1, 1)
        plt.plot(np.cumsum(rewards))
        plt.plot(np.cumsum(rewards_opt))
        plt.legend(['Charge Immediately', 'Optimal'])
        plt.ylabel('Cumulative reward')
        # plt.xticks(np.arange(0, steps, 1))
        plt.title('Cumulative reward')

        # Plot the reward per step in subplot 2
        plt.subplot(2, 1, 2)
        plt.plot(rewards)
        plt.plot(rewards_opt)
        plt.legend(['Charge Immediately', 'Optimal'])
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        # plt.xticks(np.arange(0, steps, 1))
        plt.title('Reward per time step')

        plt.tight_layout()
        plt.savefig(f'plots/{env.sim_name}/RewardsComparison.html',
                    format='svg', dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    eval()
