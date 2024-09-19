# Performance Boosting Controllers

## Code structure 

- Controllers:
  - PB_Controllers: Code for the performance boosting controller
  - Contractive REN: Code for the REN implementation
 
- Experiments
  - Saved results: Folder containing the output plots
  - run.py: Main file, to run for the simulations
  - arg_parsers.py: Main arguments, to modify from the command line if needed
 
- Loss
  - DHN loss: Loss function specifications for the optimization problems
 
- Plants
  - DHN_SYS: Dynamics and rollout of the system
  - DHN_Dataset: Generation of the data to train on 
