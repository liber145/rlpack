TRPO
====

TRPO是一种经典的强化学习算法，全称是Trust Region Policy Optimization，中文译为信赖域策略优化。 策略梯度算法更新策略时，如何选择合适步长从而确保累积奖励增加是一个关键问题。
TRPO通过限制新策略在旧策略的邻域中搜索，具有

- 累积奖励递增的理论分析，
- 不错的训练效果。


优化目标
------------

策略 :math:`\pi` 的累积奖励定义为 :math:`J(\pi) = \mathbb{E}_{s_0, a_0, ... \sim \pi} \sum_{t=0}^\infty \gamma^t r(s_t, a_t)`.
Sham Kakade（2012）分析了两个策略 :math:`\tilde{\pi}` 和 :math:`\pi` 之间的累积奖励差值，

.. math::

    J(\tilde{\pi}) - J(\pi) &= \mathbb{E}_{s_0,a_0, ... \sim \tilde{\pi}} \sum_{t=0}^\infty \gamma^t A_\pi(s_t, a_t) \\
    &= \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) A_\pi(s, a).


其中，:math:`A_\pi(s_t, a_t)` 表示优势函数，:math:`A_\pi(s_t, a_t) = Q_\pi(s_t, a_t) - V_\pi(s_t)`.


因此，给定当前策略 :math:`\pi` ，我们可以通过提升差值项来改进模型。
在实际计算过程中，动作分布 :math:`\tilde{\pi}(a|s)`可以通过重要性采样（importance sampling）解决，

.. math::

    \sum_a \pi(a|s) \frac{\tilde{\pi}(a|s)}{\pi(a|s)} A_\pi(s, a),

但状态分布 :math:`\rho_{\tilde{\pi}}(s)` 难以通过重要性采样解决，因为状态分布受决策序列影响，概率依赖很深。
TRPO使用旧策略对应的状态分布 :math:`\rho_{\pi}(s)` 去近似该状态分布。
因此，优化目标转化为最大化下面的近似累积奖励差函数，

.. math::
    L_\pi(\tilde{\pi}) = \sum_s \rho_\pi(s) \sum_a \pi(a|s) \frac{\tilde{\pi}(a|s)}{\pi(a|s)} A_\pi(s,a)


以上优化目标和普通Actor Critic的优化目标是相同的。可见，普通Actor Critic也有近似优化目标。
TRPO进一步添加了KL散度来约束策略更新，最终的优化目标为，

.. math::
    & \max_{\tilde{\pi}} \sum_s \rho_\pi(s) \sum_a \pi(a|s) \frac{\tilde{\pi}(a|s)}{\pi(a|s)} A_\pi(s,a) \\
    & s.t. ~~~~ \mathbb{E}_{s \sim \rho_\pi} D_{KL}(\pi(\cdot|s) \| \tilde{\pi}(\cdot|s)) \leq \epsilon \nonumber




理论分析
---------


优化近似的目标会有两个问题

- 不知道更新方向对不对，
- 不知道如何挑选合适的步长。


TRPO建立了以下的边界分析，

.. math::
    J(\tilde{\pi}) - J(\pi) \geq L_\pi(\tilde{\pi}) - CD_{KL}^\max(\pi, \tilde{\pi}) \\
    \text{其中，} C= \frac{4\gamma \epsilon}{(1-\gamma)^2}, \epsilon = \max_{s,a} |A(s,a)|.

以上不等式打通了累积奖励增益 :math:`J(\tilde{\pi}) - J(\pi)` 和近似目标 :math:`L_\pi(\tilde{\pi})` 之间的关系。
由此，我们不需要担心上述两个问题，只需优化不等式右边的项。
注意，具体求解优化目标时，我们进一步近似了策略约束，将KL散度的最大化操作替换为平均操作。


计算过程
-----------

求解优化问题时，我们对目标进行一阶泰勒近似，得到

.. math::
    \mathbb{E}_{s \sim \rho_{\pi_{\theta_{old}}}(\cdot), a \sim \pi_{\theta_{old}}(\cdot|s)} \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\pi_{\theta_{old}}}(s,a)
    = g^\top (\theta - \theta_{old}) + K_0,

其中 :math`g` 表示 :math:`A_{\pi_{\theta_{old}}}(s,a) \pi_\theta(a|s) / \pi_{\theta_{old}}(a|s)` 在 :math:`\theta = \theta_{old}` 处导数的期望，
:math:`K_0` 表示和 :math:`\theta` 无关的常数。
我们对策略约束使用二阶泰勒近似，可以得到

.. math::
    \mathbb{E}_{s \sim \rho_{\pi_{\theta_{old}}}(\cdot)} D_\alpha (\pi_{\theta_{old}}(\cdot|s) \| \pi_\theta(\cdot|s)) 
    = \frac{1}{2} (\theta - \theta_{old})^\top H (\theta - \theta_{old}) + K_1,


其中 :math:`H` 表示在等式左边项在 :math:`\theta=\theta_{old}` 处的二阶导数，:math:`K_1` 表示和 :math:`\theta` 无关的常数。
注意，上面等式的右边没有一阶项，这是因为左边项在 :math:`\theta = \theta_{old}` 的一阶项为零。
在实现过程中，上述一阶导数和二阶导数期望的计算都是使用采样的数据近似计算得到的。


我们去掉与 :math:`\theta` 无关的常数项之后，可以得到如下的优化问题，

.. math::
    & \min_\theta ~   - g^\top (\theta - \theta_{old}) \\
    & s.t. ~~ \frac{1}{2}(\theta - \theta_{old})^\top H (\theta - \theta_{old}) \leq \epsilon.


上式可以转化成等价的最小最大问题，

.. math::
    \min_\theta  \max_{\lambda \geq 0} ~  L(\theta, \lambda) = - g^\top(\theta - \theta_{old}) + 
    \lambda \cdot  (\frac{1}{2} (\theta - \theta_{old})^\top H (\theta - \theta_{old}) - \epsilon). 

接下来我们使用KKT条件求解上述问题。
根据 :math:`L(\theta, \lambda)` 的稳定性，我们可以得到 :math:`\partial L/\partial \theta = 0`，
进而推导出 :math:`\theta = \theta_{old} + \lambda^{-1} H^{-1}g`.
然后我们将其带入到  :math:`\partial L/\partial \lambda = 0` ，
可以计算得到 :math:`\lambda = \sqrt{ (g^\top H^{-1} g)/(2\epsilon) }`.
从而可以计算得出问题的解 :math:`\theta = \theta_{old} + \sqrt{ 2\epsilon (g^\top H^{-1}g)^{-1} } H^{-1}g`.




参考文献
---------

[1] Schulman, John, et al. "Trust region policy optimization." International Conference on Machine Learning. 2015.
