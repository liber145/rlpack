DQN
===


DQN is an off-policy algorithm.


Quick Review
------------

DQN lighted the fire of reinforcement learning.
It introduces deep learning to Q-learning and proposes two key ideas to make the learning not divergent [1].

The optimization objective of DQN can be formated by:

.. math::
   L_i(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \left[ \left(r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s,a; \theta_i) \right)^2  \right]


The two key ingredients are in the above equation:

1. :math:`U(D)` means to uniform sample experienced transitions :math:`(s, a, r, s')` from an experience replay buffer :math:`D`. This alleviates the correlations in the observed sequence and smoothes over changes in the data distribution.
2. :math:`Q(s', a'; \theta_i^-)` is a target action value function, which helps reducing correlations with the target.


   
Reference
---------

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529.
