# MPC-G2V-V2G
Open-source Model Predictive Control implementations. [Paper Link](https://arxiv.org/abs/2405.11963)

## Installation

Make sure you installed the [EV2Gym](https://github.com/StavrosOrf/ev2gym) first.

```bash
pip install ev2gym
```

Run a simple example with MPC:
```python

from ev2gym.models.ev2gym_env import EV2Gym
from occf_mpc import OCMF_V2G, OCMF_V2G
from eMPC import eMPC_V2G, eMPC_G2V

# path to the configuration file
config_file = "V2G_MPC.yaml"

# create the simulator environment
env = EV2Gym(config_file=config_file,
                    verbose=True,
                    save_replay=True,                       
                    save_plots=True,
                    )

state, _ = env.reset()
agent = eMPC_V2G(env, control_horizon=25, verbose=False)
# run the simulation
for t in range(env.simulation_length):        
    # get the action from the MPC algorithm
    actions = agent.get_action(env)
    # Step the simulation
    new_state, reward, done, _, stats = env.step(
        actions, visualize=True)  # takes action        
    
print(stats)

```




## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@misc{diazlondono2024simulationtoolv2genabled,
      title={A Simulation Tool for V2G Enabled Demand Response Based on Model Predictive Control}, 
      author={Cesar Diaz-Londono and Stavros Orfanoudakis and Pedro P. Vergara and Peter Palensky and Fredy Ruiz and Giambattista Gruosso},
      year={2024},
      eprint={2405.11963},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2405.11963}, 
}
```


