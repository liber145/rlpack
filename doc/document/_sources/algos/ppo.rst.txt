PPO
===

PPO is an on-policy algorithm.


Quick Review
------------

While TRPO uses a hard constraint to control the policy update in a small scale,
PPO modifeies the surrogate objective by clipping the probability ratio [1].

.. math::
    L(\theta) &= \mathbb{E}_t \min( r_t(\theta) A_t, clip( r_t(\theta) A_t , 1-\epsilon, 1+\epsilon) )  \\
    r_t(\theta) &=  \frac{\pi_{\theta}(a_t | s_t)}{ \pi_{\theta_{old}}(a_t | s_t) }
   

Practice
--------

To update the policy for PPO, one only needs to call a gradient descent optimizer.
Its implementation is quite easier than TRPO.
In practice, PPO is more robust than TRPO.
The champion of OpenAI Retro Contest uses an A3C version of PPO, join PPO `link <https://blog.openai.com/first-retro-contest-retrospective/>`_.


Reference
---------

[1] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
