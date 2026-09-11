# Lyapunov Stable Graph Neural Flow
This repository contains the code for the paper “Lyapunov Stable Graph Neural Flow”.

Requirements

To install the required dependencies, refer to the environment.yaml file

Reproducing Results

The training process of FL-GNN contains two stages. In the first stage, we train the base dynamical system until convergence and then freeze the network parameters. In the second stage, we focus on training the fractional Lyapunov stability module.


For table2，\
1.1 src-process-for-table2: used to  generate the adversarial graphs

1.2 src-table2: used to generate the experiments in Table2

For table3,\
1 src-table3: used to generate the experiments in Table3


Our code is developed based on the following repos:

The GIA attack method is based on the [GIA-HAO](https://github.com/LFhase/GIA-HAO/tree/master) repo.  
The HANG model is based on the [GraphCON](https://github.com/tk-rusch/GraphCON) framework.  
The METATTACK and NETTACK methods are based on the [deeprobust](https://github.com/DSE-MSU/DeepRobust) repo.



