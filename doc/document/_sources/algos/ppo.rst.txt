PPO
===

PPO全称是Proximal Policy Optimization，中文译为近端策略优化。
PPO简化了TRPO中复杂的计算流程，从而降低了计算复杂度以及实现难度。


优化目标
----------


PPO简化了TRPO中的优化问题，将优化问题转化为，

.. math::
    \max_\theta \mathbb{E}_{s \sim \rho_{\pi_{\theta_{old}}}(\cdot), a \sim \pi_{\theta_{old}}(\cdot|s)} \min \left(  \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\pi_{\theta_{old}}}(s,a), clip \left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} , 1-\epsilon, 1+\epsilon \right) A_{\pi_{\theta_{old}}}(s,a)\right)

沿用TRPO中的思路，将新策略约束在旧策略的邻域内：首先使用clip操作，约束新旧策略在动作概率上的比率，获得一个近似目标；
然后通过min操作，确保最终的优化目标是一个真实目标的下界。
最后，求解优化问题来抬高下界，从而达到改进目标的效果。


直观释义
--------------


优势函数的定义是 :math:`A_{\pi_{old}}(s,a) = Q_{\pi_{old}}(s,a) - V_{\pi_{old}}(s)` ，
表示采样动作相对于平均动作的优势值。

- 当 :math:`A > 0` 时，表示此时优势值为正，即当前策略在该状态上正确执行，没必要在此样本上过度修正算法。
因此，min操作和clip操作组合使得如果比值超过 :math:`1+\epsilon` ，最终为 :math:`1+\epsilon` ，否则保持原值。
这样就限制了更新程度。
- 当 :math:`A < 0` 时，表示此时优势值为负（:math:`-\max(r, clip(r, 1-\epsilon, 1+\epsilon))A` ，:math:`r` 表示比值），即当前策略在该状态上效果不好，有必要在此样本上修正算法。
因此，min操作和clip操作组合使得如果比值低于 :math:`1-\epsilon`，最后为 :math:`1-\epsilon` ，否则保持原值。
这样就使得更新成都可以很大。



参考文献
---------

[1] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
