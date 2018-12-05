TRPO
====

TRPO is an on-policy algorithm [1].
It uses KL-divergence to control the updated policy not exceeding a reasonable ball of the last policy.


Quick Review
------------

Like all reinforcement learing algorithms, TRPO cares the future accumulative reward:

.. math::

   \eta(\pi) = \mathbb{E}_{s_0, a_0, ...} \sum_{t=0}^\infty \gamma^t r(s_t).


Note that the expected return of another policy :math:`\eta(\tilde{\pi})` can be expressed:

.. math::

   \eta(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A_\pi(s, a)

   
Therefore, given the current policy :math:`\pi`, to update the new policy :math:`\tilde{\pi}`, we will maximize the right term of the above equation.
However, it is not easy to solve since we do not have :math:`\rho_{\tilde{\pi}}(s)`.
Instead, Consider the surrogate function:

.. math::

   L_{\pi}(\tilde{\pi}) = \eta(\pi) + \sum_s \rho_{\pi} (s) \sum_a \tilde{\pi}(a|s) A_\pi(s, a).


Further, [1] gives the relationship between new policy and old policy.

.. math::

   & \eta(\pi_{new}) \geq L_{\pi_{old}}(\pi_{new}) - CD_{KL}^{max} (\pi, \tilde{\pi}) \\
   & \text{where}~~ C = \frac{4\epsilon \gamma}{ (1-\gamma)^2 } 

   
[1] considers solving the below optimization problem:

.. math::

   &\max_{\theta} L_{\theta_{old}}(\theta) \\
   &\text{subject to} ~~ D_{KL}^{max} (\theta_{old}, \theta) \leq \delta


The maximum KL divergence is depended on every point in the state space and is impractical to estimate.
Instead, a heuristic approximation is used:

.. math::

   \bar{D}_{KL}^{\rho} (\theta_1, \theta_2) = \mathbb{E}_{s\sim \rho} D_{KL}( \pi_{\theta_1}(\cdot|s) \| \pi_{\theta_2(\cdot | s)} )


Computation
-----------
   
To sovle it, [1] uses the gradient descent algorithm.

.. math::

   &\min_x \nabla f(x) (x - x_0)   \\
   &\text{s.t.}~~   (x - x_0)^\top H (x - x_0) \leq \delta


The Lagrangian function can be expressed by:   

.. math::

   L(\lambda, x) \nabla f(x) (x - x_0) + \lambda [(x - x_0)^\top H (x - x_0) - \delta]

Therefore, we can get the update direction :math:`d(x) = H^{-1} \nabla f(x)`.
Then, we can use line search to get a proper step length such that the new policy satisfies the KL-divergence constraint.


Reference
---------

[1] Schulman, John, et al. "Trust region policy optimization." International Conference on Machine Learning. 2015.
