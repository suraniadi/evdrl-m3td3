# Competitive Reinforcement Learning for Real-Time Pricing and Scheduling Control in Coupled EV Charging Stations and Power Networks

PyTorch implementation of Minimax Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (M3TD3). 

Networks are trained using [PyTorch 1.11](https://github.com/pytorch/pytorch) and Python 3.9.16. We used NVIDIA's RTX 2060 GPU for the 150,000 iterations training. 

### Usage
The paper results can be reproduced by running the `main.ipynb` with varying parameter settings for the plotting functions.

The non-averaged graphs can be displayed using the bottom cells of the same notebook.

### Results
Code is no longer exactly representative of the code used in the paper. Minor adjustments to hyperparamters, etc, to improve performance. Learning curves are still the original results found in the paper.

Code partially adapted from the following [paper](https://arxiv.org/abs/1802.09477).
