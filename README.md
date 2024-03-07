# MPC-G2V-V2G
Open-source MPC implementation for EV V2G smart charging for the [EVs-Simulator](https://github.com/StavrosOrf/EVsSimulator) from paper " ... ".

## Installation

Make sure you installed the [EVs-Simulator](https://github.com/StavrosOrf/EVsSimulator) first.

```bash
pip install EVsSimulator
```

Run a simple example with MPC:
```python

from EVsSimulator.ev_city import EVsSimulator
from occf_mpc import OCCF_V2G, OCCF_G2V
from eMPC import eMPC_V2G, eMPC_G2V

# path to the configuration file
config_file = "V2G_MPC.yaml"

# create the simulator environment
env = EVsSimulator(config_file=config_file,
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
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```


