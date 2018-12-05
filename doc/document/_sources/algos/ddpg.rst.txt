DDPG
====

DDPG is an off-policy algorithm.


Quick Review
------------

DDPG is the deep learning vergion of deterministic policy gradient (DPG) algorithm [2].
DPG consider policy gradient algorithm in the context of deterministic policy.

Simliar to policy gradient theorem, [2] gives a deterministic policy gradient theorem,

.. math::
    \nabla_\theta J(\mu_\theta) &= \sum_s \rho^\mu(s) \nabla_\theta \mu_\theta (s)  \nabla_a Q^\mu(s,a)|_{a = \mu_\theta(s)} \mathrm{d} s \\
    &= \mathbb{E}_{s \sim \rho^\mu}  \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a)|_{a = \mu_\theta(s)}

The action value udpate is to minimize TD error between target value and current value as usual.


Implementation
--------------

The policy update can be rewritten to :math:`\nabla_\theta Q(s, \mu_\theta(s))`.
We can write the policy loss as :math:`\mathbb{E}_s [-Q(s, \mu_\theta(s))]`, then pick an optimizer to do gradient descent on policy loss and value loss iteratively.


Given a state, straightforward action inference by old policy makes no exploration.
[1] introduces an Ornstein-Uhlenbeck process to generate temporally correlated exploration.



Reference
---------

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

[2] Silver, David, et al. "Deterministic policy gradient algorithms." ICML. 2014.
