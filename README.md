# Reinforcement_learning
## Abstract 
We aim to determine the optimal trajectory of a smart micro-swimming particle in an environment with potential gradients, employing calculus equations
and reinforcement learning techniques. Initially, we apply these methods to a Mexican hat potential without brim, then increase the complexity of the
potential function.For lower potential values, the particle can easily traverse the potential barrier, whereas for higher potential values, circumventing
the barrier proves to be optimal. However, as the grid resolution increases, the brute force solution becomes exponentially time-consuming.The
complexities of the potential functions pose computational challenges when using basic calculus methods, prompting us to employ reinforcement
learning. We utilize the q-learning method to obtain results. Initially, we apply q-learning to solve the problem within the Mexican hat potential and then
extend it to more complex potentials with hills and wells. Although Q-learning offers improved time complexity, it struggles to adapt to changes in the
environment.To address this limitation, we turn to deep reinforcement learning techniques to train the agent to perform optimally even in novel potential
environments, thereby aiming to generalize the policy guiding the agent’s actions
