A2C
===

Advantage Actor Critic is an off-policy algorithm.


Quick Review
------------

First, let's look at REINFORCE algorithm, which is a Monte Carlo policy gradient aglrothm.

.. math::
   \nabla \eta(\theta) &= \sum_s d_\pi(s) \sum_a q_\pi(s, a) \nabla_\theta \pi(a | s; \theta) \\
   &= \mathbb{E}_{(s_t, a_t) \sim \pi} \gamma^t q_\pi(s_t, a_t) \nabla_\theta \pi(a_t | s_t; \theta)

   
REINFORCE algorithm uses Monte Carlo method to estimate the expected :math:`q(s_t, a_t)`.

Note that :math:`\mathbb{E}_{(s_t, a_t) \sim \pi} b(s_t) \nabla_\theta \pi(a_t | s_t; \theta) = 0`, we have   

.. math::
   \nabla \eta(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi} \gamma^t [q_\pi(s_t, a_t) - b(s_t)] \nabla_\theta \pi(a_t | s_t; \theta)
   
The term :math:`b(s_t)` called *baseline* is usually estimated by state value :math:`v(s_t)`.
The residual term :math:`q(s_t, a_t) - v(s_t)` is called *advantage*.
In general, the baseline leaves the expected value of the update unchanged, but it can have a large effect on reducing its variance [1].


Now, let's go to advantage actor critic (A2C).
Instead, A2C uses a state value approximate function to estimate :math:`v(s_t; \theta)`.
The action value can be derived as :math:`q(s_t, a_t) = r_t + \gamma v(s_{t+1}; \theta)`.
The critic part updates the value function from TD error.
The actor part updates the policy function by policy gradient.



Reference
---------

[1] Sutton, Richard S., and Andrew G. Barto. "Reinforcement Learning: An Introduction." (1998).
